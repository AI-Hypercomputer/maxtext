# Integrative RAG Agent

## Overview

The Integrative RAG Agent automates the end-to-end process of extracting Python code blocks, generating semantic embeddings, identifying model components, organizing them in hierarchical order, and finally performing large-scale code conversion. This agent integrates scraping, RAG (Retrieval-Augmented Generation), and structured code transformation workflows into a unified pipeline.

It is designed to work with large model implementations from repositories like Hugging Face `transformers`, enabling developers to prepare complex models for conversion, testing, and debugging.

## Workflow

1. **Scrape Python Blocks**  
   Extract all Python code blocks defined in `config.py` from the MaxText repository using `scrap_all_python_blocks.py`.

2. **Generate Descriptions and Embeddings**  
   Run `llm_rag_embedding_generation.py` to generate semantic descriptions and embeddings for all extracted code blocks. The embeddings are stored in `rag_store.db`.

3. **Get Model File and Class Info**  
   Use `get_model_info.py` with a Hugging Face `model-id` to fetch the target model’s file path and class name.

4. **Sort Components in Hierarchical Order**  
   Use `sort_components_in_hierarchical_order.py` to analyze the target file and produce a dependency graph in JSON format. This ensures modules are ordered correctly for conversion.

5. **Review Dependencies**  
   Developers should review the generated JSON (`*_files_order.json`) to confirm or adjust module dependencies. Any removed modules must also be cleaned from dependency references.

6. **Run Code Conversion**  
   Execute `llm_code_conversion.py` with the target module name. This step generates converted code for each module in the hierarchy and saves them into `MaxText/experimental/agent/{Module_Name}` for further review and refinement.

## File Descriptions

- **`scrap_all_python_blocks.py`**: Scrapes Python blocks from `config.py` in MaxText repo.  
- **`llm_rag_embedding_generation.py`**: Generates semantic descriptions and embeddings for each scraped block, storing results in `sag_store.db`.  
- **`get_model_info.py`**: Retrieves the class name and file path for a model from Hugging Face by its `model-id`.  
- **`sort_components_in_hierarchical_order.py`**: Builds a dependency hierarchy of model components and outputs a JSON ordering file.  
- **`llm_code_conversion.py`**: Converts all listed modules from the hierarchy JSON into their target format and saves them for manual review.  

## Setup

1. **Install Dependencies**  
   ```bash
   pip install python-dotenv google-generativeai backoff google-genai requests
   ```

2. **Configure Environment Variables**  
   Create a `.env` file in the root directory with the required credentials:  
   ```env
   GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   Model="gemini-2.5-pro"
   GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
   ```

3. **Prepare Data**  
   Ensure your Hugging Face model files are accessible, and set paths accordingly in the scripts.

## Usage

#### Steps
1.**Run the script to scrape Python blocks**

```python scrap_all_python_blocks.py```

This will scrape all Python blocks defined in config.py from the MaxText repo.

2.**Generate descriptions and embeddings**

```python llm_rag_embedding_generation.py```

This will generate descriptions for all MaxText code blocks scraped in step 1 and create embeddings for them. The results will be saved in the sag_store.db database.

3.**Get the file and class name for the model you want to convert**
Copy the model ID from Hugging Face and run:
   
    ```python get_model_info.py --model-id <model-id>```

   Example:

   ```python get_model_info.py --model-id Qwen/Qwen3-235B-A22B-Thinking-2507-FP8```

   This will return the class name and file path from transformers. and these will be used in following step as filepath and  module_name. You need to use this infomation to the step 4.


4.**Sort components in hierarchical order**

```python sort_components_in_hierarchical_order.py --entry-file-path <filepath> --entry-module <module_name>```

Example:

```python sort_components_in_hierarchical_order.py --entry-file-path transformers/models/qwen3_moe/modeling_qwen3_moe.py --entry-module Qwen3MoeForCausalLM```

This will generate a file Qwen3MoeForCausalLM_files_order.json in the results/ folder.
The file contains the list of modules for Qwen3MoeForCausalLM along with their dependencies.

5.**Review and adjust module dependencies**
The developer should review the generated JSON file and add or remove modules as needed.
If a module is removed, its references must also be removed from other dependency entries.

6.**Run the code conversion**

```python llm_rag_code_conversion.py --module-name <module_name>```

Example:

```python llm_rag_code_conversion.py --module-name Qwen3MoeForCausalLM```

requests  This will convert all modules listed in Qwen3MoeForCausalLM_files_order.json and save them in:
MaxText/experimental/agent/{Module_Name}

 _The developer must then review these files one by one and fix any errors._

## Output

- **`rag_store.db`**: Embedding database storing semantic representations of Python blocks.  
- **`*_files_order.json`**: Ordered list of components and dependencies for the target model.  
- **Converted Code Directory**: Contains generated files for the specified module, ready for developer review and debugging.  