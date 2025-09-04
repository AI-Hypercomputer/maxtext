# Code Generation Agent

This agent is designed to automate the conversion of Python code, with a primary focus on transforming PyTorch code into functionally equivalent JAX code using the Google Gemini model.

The workflow is typically a two-step process:
1.  **Test File Generation**: Use `make_pytorch_file.py` to analyze a large, complex Python file (e.g., a model implementation from a library like `transformers`) and extract standalone, self-contained PyTorch modules. This step is crucial for creating manageable test cases for the conversion agent.
2.  **Code Conversion**: Use `llm_code_generation.py` to take the generated PyTorch files (or any other PyTorch file) and convert them into JAX.

## File Descriptions

-   **`llm_agent.py`**: This file contains the `GeminiAgent` class, which serves as a robust interface to the Google Gemini API. It handles model initialization, API call configuration, and implements a backoff/retry mechanism to manage transient network errors.

-   **`prompt_code_generation.py`**: This file stores the prompt templates that guide the Gemini model. It includes a `SystemPrompt` that defines the model's role as an expert ML engineer specializing in code conversion, and a `CODE` template for formatting the user's request.

-   **`make_pytorch_file.py`**: This utility script is designed to create test files for the conversion agent. It parses a given Python source file, identifies PyTorch-related classes (e.g., those inheriting from `torch.nn.Module`) and functions that are "standalone" — meaning they do not have dependencies on other code defined within the same file. These extracted modules are then saved as individual Python files, ready to be used as input for the conversion process.

-   **`llm_code_generation.py`**: This is the main script for performing the code conversion. It can process a single Python file or an entire directory of files. It reads the source PyTorch code, combines it with the prompts from `prompt_code_generation.py`, sends the request to the `GeminiAgent`, parses the JAX code from the response, and saves it to an output file.

## Setup

1.  **Install Dependencies**:
    Make sure you have the required Python packages installed.
    ```bash
    pip install python-dotenv google-generativeai backoff
    ```

2.  **Configure Environment Variables**:
    Create a `.env` file in the `code_generation_agent` directory with your Google API key and the desired model name.
    ```.env
    # .env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    Model="gemini-1.5-pro-latest"
    ```

3.  **Set Output Directory**:
    In `llm_code_generation.py`, the `JAX_OUTPUT_DIR` variable is configured to save converted files into `code_generation_agent/dataset/jax_converted/`. You can modify this variable if you need a different output path.
    ```python
    # in llm_code_generation.py
    JAX_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset/jax_converted")
    ```

## Usage

To run the agent scripts, you should first navigate to the `src/MaxText/experimental/agent` directory. The scripts are designed to be run as Python modules from this location.

### Step 1: Create Standalone PyTorch Test Files

This step is for breaking down large, complex PyTorch files into smaller, independent modules that are suitable for conversion.

**Example Invocation:**

This example command analyzes the Llama model file from the Hugging Face `transformers` repository and extracts standalone modules. Run it from the `src/MaxText/experimental/agent` directory.

```bash
python3 -m code_generation_agent.make_pytorch_file \
  --entry_file_path "transformers/models/llama/modeling_llama.py" \
  --base_path "https://github.com/huggingface/transformers/blob/main/src"
```

The extracted files will be saved in the `dataset/Pytorch/` directory (as configured in `make_pytorch_file.py`). Each filename will be a combination of the original filename and the extracted module name (e.g., `modeling_llama__LlamaDecoderLayer.py`).

### Step 2: Convert PyTorch Code to JAX

Once you have your PyTorch files (either created from Step 1 or from another source), you can use `llm_code_generation.py` to convert them.

**Convert a Single File:**

```bash
python llm_code_generation.py --file dataset/PyTorch/modeling_llama__LlamaDecoderLayer.py 
```

**Convert All Files in a Folder:**

```bash
python llm_code_generation.py --folder dataset/PyTorch/
```

The converted JAX files will be written to the directory specified by `JAX_OUTPUT_DIR` in `llm_code_generation.py`.
