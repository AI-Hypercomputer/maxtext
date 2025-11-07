# Integrative RAG Agent

## Setup

1. **🚀 Setup and Installation**
   ```bash
   # 1. Clone the repository
   git clone https://github.com/AI-Hypercomputer/maxtext.git
   cd maxtext

   # 2. Checkout the development branch
   git checkout jennifer/pytorch_jax_migration

   # 3. Create virtual environment
   uv venv --python 3.12 --seed maxtext_venv
   source maxtext_venv/bin/activate

   # 4. Install dependencies in editable mode
   pip install uv
   # install the tpu package
   uv pip install -e .[tpu] --resolution=lowest
   # or install the gpu package by running the following line
   # uv pip install -e .[cuda12] --resolution=lowest
   install_maxtext_github_deps

   ```

2. **Configure Environment Variables**  
    Create a `.env` file in the current folder with the required credentials:  
    ```env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    Model="gemini-2.5-pro"
    GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
    ```

    Alternatively, to speed up the process, try the gemini flash lite model 
    without reasoning.
    ```env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    Model="gemini-2.5-flash-lite"
    GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
    ```
## Usage
cd to the folder `maxtext/src/MaxText/experimental/agent/integrative_rag_agent`

#### Steps
1.**Run the script to scrape Python blocks**

```python scrap_all_python_blocks.py```

This will scrape all Python blocks defined in config.py from the MaxText repo.
This will create embedding_cache.db and maxtext_blocks.json under the dataset folder.

2.**Generate descriptions and embeddings**

```python llm_rag_embedding_generation.py```

This will generate descriptions for all MaxText code blocks scraped in step 1 and create embeddings for them. The results will be saved in the rag_store.db database.

This step takes ~2 hours with gemini-2.5-pro, takes ~15 minutes with gemini-2.5-flash-lite. To skip this and use the pre-generated files by gemini-2.5-pro, copy the dataset folder from examples/dataset directly into the integrative_rag_agent/ folder.

The final file structure should look like this:

```
integrative_rag_agent/
├── dataset/
│   ├── rag_store.db
│   ├── maxtext_blocks.json
│   └── maxtext_blocks_description.json
├── llm_rag_embedding_generation.py
└── ... (other agent files)
```

Once completed, this step and step1 need not be repeated every time you convert new code. You can reuse the blocks and descriptions generated in steps 1 and 2, unless Maxtext has a recent change that updates the blocks.

3.**Get the file and class name for the model you want to convert**
Copy the model ID from Hugging Face and run:
   
    ```python get_model_info.py --model-id <model-id>```

   Example:

   ```python get_model_info.py --model-id Qwen/Qwen3-235B-A22B-Thinking-2507-FP8```

   This will return the class name and file path from transformers. and these will be used in following step as filepath and module_name. You need to use this information in step 4.


4.**Sort components in hierarchical order**

```python sort_components_in_hierarchical_order.py --entry-file-path <filepath> --entry-module <module_name>```

Example:

```python sort_components_in_hierarchical_order.py --entry-file-path transformers/models/qwen3_moe/modeling_qwen3_moe.py --entry-module Qwen3MoeForCausalLM```

This will generate a file Qwen3MoeForCausalLM_files_order.json in the results/ folder.
The file contains the list of modules for Qwen3MoeForCausalLM along with their dependencies.

**Filter Modes:**

* **Standard (Default):** This script uses a `standard` LLM filter by default to speed up the process. It removes non-essential code (like logging and metrics) but correctly keeps structural code (like base classes and type hints).

* **Aggressive:** To use a faster, more aggressive filter (which may incorrectly remove base classes or type hints), use the `--filter-mode aggressive` flag:

```
python sort_components_in_hierarchical_order.py ... --filter-mode aggressive
```

* **None:** To disable the llm dependency filter and process the entire dependency list (which can take 6+ hours and risks running out of memory), use the `--filter-mode none` flag:
```
python sort_components_in_hierarchical_order.py ... --filter-mode none
```

* **Note: Skip this step (Optional)**
To skip this analysis and use the example output, copy the results folder from `examples/results` directly into the `integrative_rag_agent/` folder.

The final file structure should look like this:
```
integrative_rag_agent/
├── results/
│   └── Qwen3MoeForCausalLM_files_order.json
├── sort_components_in_hierarchical_order.py
└── ... (other agent files)
```

5.**Run the code conversion**

```python llm_rag_code_conversion.py --module-name <module_name>```

Example:

```python llm_rag_code_conversion.py --module-name Qwen3MoeForCausalLM```

requests  This will convert all modules listed in Qwen3MoeForCausalLM_files_order.json and save them in:
MaxText/experimental/agent/{Module_Name}

 _The developer must then review these files one by one and fix any errors._

## Common debugging steps

1. **If Gemini api is using a free tier key and run out of quota**  
If Gemini api is under free tier and run out of quota, but you are not using a free tier account, try refresh your api key:

```
unset GOOGLE_API_KEY
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```