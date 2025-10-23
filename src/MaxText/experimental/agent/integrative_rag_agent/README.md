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
   Create a `.env` file in the root directory with the required credentials:  
   ```env
   GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   Model="gemini-2.5-pro"
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

This will generate descriptions for all MaxText code blocks scraped in step 1 and create embeddings for them. The results will be saved in the sag_store.db database.

3.**Get the file and class name for the model you want to convert**
Copy the model ID from Hugging Face and run:
   
    ```python get_model_info.py --model-id <model-id>```

   Example:

   ```python get_model_info.py --model-id Qwen/Qwen3-235B-A22B-Thinking-2507-FP8```

   This will return the class name and file path from transformers. and these will be used in following step as filepath and  module_name. You need to use this information in step 4.


4.**Sort components in hierarchical order**

```python sort_components_in_hierarchical_order.py --entry-file-path <filepath> --entry-module <module_name>```

Example:

```python sort_components_in_hierarchical_order.py --entry-file-path transformers/models/qwen3_moe/modeling_qwen3_moe.py --entry-module Qwen3MoeForCausalLM```

This will generate a file Qwen3MoeForCausalLM_files_order.json in the results/ folder.
The file contains the list of modules for Qwen3MoeForCausalLM along with their dependencies.
By default, this use llm to filter out caching/checkpoints, debug code and etc. To disable llm filter (warning: this will take much longer) and reveal the entire dependency order, use the `--disable-llm-filter` flag

Example:

```python sort_components_in_hierarchical_order.py --entry-file-path transformers/models/qwen3_moe/modeling_qwen3_moe.py --entry-module Qwen3MoeForCausalLM --disable-llm-filter```


5.**Run the code conversion**

```python llm_rag_code_conversion.py --module-name <module_name>```

Example:

```python llm_rag_code_conversion.py --module-name Qwen3MoeForCausalLM```

requests  This will convert all modules listed in Qwen3MoeForCausalLM_files_order.json and save them in:
MaxText/experimental/agent/{Module_Name}

 _The developer must then review these files one by one and fix any errors._

## Common debugging steps

1. **If Gemini api is under free tier and run out of quota**  
If Gemini api is under free tier and run out of quota, but you are not using a free tier account, try refresh your api key:

```
unset GOOGLE_API_KEY
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```