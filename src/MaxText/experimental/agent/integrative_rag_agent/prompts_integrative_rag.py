# Copyright 2023–2025 Google LLC
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
This file contains prompts and templates used by the Integrative RAG Agent for code analysis, 
description generation, and module naming.
"""

Description_Prompt = """
  You are an expert code analyst. Your task is to analyze the provided Python code block and generate a detailed JSON description of it.

  **Context:**
  Here is the full content of the file for context:
  ```python
  {full_code_context}
  Code Block to Analyze:

  python
  Copy
  Edit
  {code_block}
  Instructions:
  Based on the code block and its context, generate a JSON object with the exact following structure. Do NOT output any text, markdown, or code formatting before or after the JSON object.

  JSON Schema:
  {{
  "module_type": "A descriptive snake_case name for the module type (e.g., vision_transformer_encoder, multi_head_attention).",
  "purpose": "A concise sentence explaining what the module does.",
  "input": {{
  "shape": "Describe the expected input tensor shape (e.g., [batch_size, sequence_length, hidden_dim]). Use 'N/A' if not applicable.",
  "dtype": "The expected data type (e.g., float32). Use 'N/A' if not applicable."
  }},
  "processing_steps": [
  "A list of key operations or method calls in the order they are applied."
  ],
  "output": {{
  "shape": "Describe the output tensor shape. Use 'N/A' if not applicable."
  }},
  "dependencies": [
  "A list of other important classes or functions this module depends on."
  ],
  "parameters": {{
  "param_name": "Description of a key parameter, often from a config object."
  }},
  "notes": [
  "Any other relevant information, such as control flow logic, specific implementation details, or assumptions."
  ]
  }}

  Special Case — If the Code Block is a Class:

  The top-level JSON should still describe the overall purpose, parameters, and dependencies of the class as above.

  Additionally, include a "methods" field that is a dictionary where each key is a method name and the value follows this schema:
  {{
  "purpose": "A concise sentence explaining what the method does.",
  "input": {{
  "shape": "Expected input tensor shape or 'N/A'.",
  "dtype": "Expected input data type or 'N/A'."
  }},
  "processing_steps": [
  "List of key operations or method calls in the order they are applied."
  ],
  "output": {{
  "shape": "Expected output shape or 'N/A'."
  }},
  "dependencies": [
  "List of other important classes/functions this method depends on."
  ],
  "notes": [
  "Additional relevant details about the method."
  ]
  }}

  Important:

  Only include information you can confidently infer from the given code and context.

  If something is not explicitly clear, use "N/A".

  Maintain consistent JSON formatting without extra commentary or markdown.

  Now, analyze the provided code block and generate the JSON object.
"""

CODE_DESCRIPTION = """
You are given:
    1. A specific code block from a file.
    2. The full code of the file the block was extracted from, for reference.
Your task is to produce a minimal, precise, and machine-readable description of the code block that:
    - Explains what the code block does in clear, concise terms.
    - Explains how the code block can be used, including its purpose, inputs, and outputs (if any).
    - Avoids unnecessary details or implementation steps that are not essential for understanding its usage.
    - Is written so that another AI agent can understand and use it.
Output Format (JSON):
{
  "functionality": "<Short, clear description of what the code block does>",
  "usage": "<Short, clear description of how to use the code block, including inputs and outputs>"
}

Code Block:

{code_block}

Full File Code (Reference):

{full_code_context}
"""


Dependency_Filter_Prompt = """
You are an expert static analysis tool. Your task is to filter a dependency list to only what is required to define and run the model's architecture.

First, think step-by-step to analyze each dependency against the rules.
Second, output *only* the final, filtered list.

Rules:
   - **KEEP** dependencies that are structurally or functionally necessary for the code to be valid. This includes:
       - Core layers and functions (e.g., nn.Linear, torch.matmul)
       - **Base classes** the model inherits from (e.g., PreTrainedModel, GenerationMixin)
       - **Type hints** used in function signatures (e.g., Unpack, Cache, Optional)
       - **Decorators** used on classes or functions (e.g., @auto_docstring)

   - **REMOVE** dependencies that are clearly non-essential "extras". This includes:
       - Caching or checkpointing utilities (unless it's a type hint like `Cache`)
       - Logging, metrics, evaluation, or visualization.
       - Debugging tools or optional framework integrations.

   - Remove unused dependencies.
   - **CRITICAL RULE: If you are not 100% sure, KEEP the dependency.** It is better to keep an unnecessary dependency than to remove an essential one.

   - Never explain anything. Do not add comments, reasoning, or descriptions.

* Output only the final list, one dependency per line.
* If the final list is empty, output the single word 'NONE'.

---
EXAMPLE 1:

Codebase:
'''python
from ...utils import auto_docstring
from ...generation import GenerationMixin
from ...utils import logging

@auto_docstring
class MyModel(GenerationMixin):
    def forward(self, ...):
        logger.info("Running forward pass")
        return ...
'''

Dependency List:
'''
transformers/utils.py#auto_docstring
transformers/generation.py#GenerationMixin
transformers/utils.py#logging
'''

Output:
'''
transformers/utils.py#auto_docstring
transformers/generation.py#GenerationMixin
'''
---

EXAMPLE 2:

Codebase:
<CODE_HERE>

Dependency List:
<DEPENDENCY_LIST_HERE>  

Output:
'''
"""

Dependency_Filter_Fast = """
You are given the source code of a project and its dependency list.
Your task is to filter the dependency list so it contains only those dependencies required to recreate the original model architecture.

Rules:
    - Keep only dependencies that are essential to define, initialize, and run the model architecture itself (e.g., core framework, math/array processing, model layers).
    - Remove any dependencies related to extra features such as:
        - caching or checkpointing systems
        - beam search or other inference strategies
        - evaluation, metrics, logging, or visualization
        - optional integrations, optimization tricks, or debugging tools

    - Remove unused dependencies.
    - Never explain anything. Do not add comments, reasoning, or descriptions.
    - If you are not sure how to filter the dependency list, output the same Dependency List exactly as given.

* Output only the final minimal list, one dependency per line, with no explanations.
* If the final list is empty, output the single word 'NONE'.

Codebase:
<CODE_HERE>

Dependency List:
<DEPENDENCY_LIST_HERE>  
"""



CODE_CONVERSION = """
You are an expert AI code translator specializing in converting PyTorch code to JAX.
I will provide you with:
    1. A code block (the part of the file I want to convert).
    2. The full source file (for reference to other functions, classes, or imports).
    3. A dictionary of existing JAX modules in the format:
        {
          "path/to/module1": "Description of what the module does",
          "path/to/module2": "Description of what the module does"
        }
    4. An example MaxText code block that demonstrates coding style, formatting, and structure conventions to follow.
    5. A dictionary of MaxText-matched dependencies that map the code block dependencies to corresponding MaxText modules, in the same {path: description} format.
    
    The keys are the paths to existing JAX module implementations, and the values are their functional descriptions.
Your tasks:
    1. Analyze the given PyTorch code block and identify all functional components it uses.
    2. Find matching modules in the provided JAX module dictionary that can replace equivalent PyTorch functionality. Use them wherever possible instead of writing new code.
    3. Only generate new JAX code for parts of the PyTorch code where no matching module exists.
    4. Ensure that the generated code:
        - Is functionally equivalent to the original PyTorch code block.
        - Uses idiomatic JAX practices (e.g., jax.numpy instead of numpy, vectorization where possible).
        - Maintains the original architecture and logic, just rewritten in JAX.
        - Preserves original function/class names unless absolutely necessary to change.
        - Matches the style and structure of the provided MaxText example code block wherever applicable (e.g., function signatures, type annotations, naming style, comments, formatting).

Output format:
    Provide only the converted JAX code block (no explanations unless asked).
    Ensure all imports are included at the top of the generated code.
    If you reused a module, add a comment showing which existing module path was used.

CODE_BLOCK:
<CODE_BLOCK>

FULL_FILE_CODE:
<FULL_FILE_CODE>

JAX_MODULES_DICT:
<JAX_MODULES_DICT>

MAXTEXT_EXAMPLE_CODE:
<MAXTEXT_EXAMPLE_CODE>


MAXTEXT_MATCHED_DEPENDENCIES:
<MAXTEXT_MATCHED_DEPENDENCIES>
"""

MODULE_NAMING_PROMPT = """You are an expert at naming Python files for new generated module.

Your task is to suggest an appropriate file name for a new module that will be saved at:
`{module_base_path}/{FileName}.py`

{existing_files_info}

Module Code:
{module_description}

Guidelines for naming:
1. Reuse whenever possible – If the functionality fits into an existing file, suggest that existing file name instead of creating a new one.
2. Group related functionality together — use broader, generic names (e.g., loss_functions, data_utils, model_layers) rather than naming after a single function.
3. Follow existing naming patterns and conventions in the directory.
4. Use descriptive, snake_case names that clearly indicate the module's purpose, but keep them general enough to hold multiple related functions.
5. Only propose a new file if there is no suitable existing file.
6. The name must be importable and follow Python naming conventions.

Output format:
Return only the suggested file name in snake_case format, no explanations or additional text.

Example outputs:
loss_functions
data_preprocessing
model_training

Suggested module name:"""
