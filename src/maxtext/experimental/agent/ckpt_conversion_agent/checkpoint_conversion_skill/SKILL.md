---
name: ckpt-conversion-agent
description: >-
  Automates the end-to-end checkpoint conversion workflow between Hugging Face and MaxText. Use this skill when orchestrating the checkpoint conversion process, which includes architecture investigation, parameter mapping, layer-by-layer transformation, full execution of conversion, and optional bidirectional conversion.
---

# Checkpoint Conversion Workflow

The automatic checkpoint conversion agent flow is a 6-phase (0-5) structured process to convert model weights between Hugging Face and MaxText, ensuring accuracy and correctness at every step.

As the primary entry point, this orchestration SKILL establishes the automated cycle and delegates execution to the appropriate phases.

## Workflow Phases

### [Phase 0: Orchestration & State Control](references/phase0_orchestration.md)
**Objective**: Entry point for eng: One SKILL to rule them all. Establish the automated cycle, connectivity, environment, execution staging, and routing.

### [Phase 1: Architecture Investigation for HF and MaxText](references/phase1_architecture.md)
**Objective**: Conduct a detailed analysis of model specifics for both Hugging Face and MaxText utilizing [graph_tracing.py](../utils/graph_tracing.py) to establish a project roadmap.

### [Phase 2: Parameter Mapping](references/phase2_parameter_mapping.md)
**Objective**: Align layer designations between Hugging Face and MaxText to confirm total parameter coverage.

### [Phase 3: Layer-by-Layer Transformation](references/phase3_layer_transformation.md)
**Objective**: Specify the necessary transformation logic for every individual layer. Pass [layer_shape_verify.py](../utils/layer_shape_verify.py) for layer shape check. Pass [layerwise_verify.py](../utils/layerwise_verify.py) for layer-by-layer forward pass check.

### [Phase 4: Execution of Conversion](references/phase4_execution.md)
**Objective**: Conduct the conversion by applying the results from Phases 2 and 3 within the consolidated checkpoint conversion framework, and run E2E verification via [forward_pass_logit_checker.py](../../../../../../tests/utils/forward_pass_logit_checker.py).

### [Phase 5: Bidirectional Conversion (Optional)](references/phase5_bidirectional.md)
**Objective**: Confirm the operational integrity of the bidirectional conversion process (from MaxText back to Hugging Face).

## Code Needed:
Here is the breakdown of the 3 files you need to modify:
### 1. [src/maxtext/checkpoint_conversion/utils/param_mapping.py](../../../../checkpoint_conversion/utils/param_mapping.py)
This file defines how MaxText handles weight paths and values compared to Hugging Face.
#### Create Mapping Function: 
Add a new function (e.g., [MODEL]_MAXTEXT_TO_HF_PARAM_MAPPING) that returns a dictionary mapping MaxText parameter keys (like params-decoder-layers_0-self_attention-query-kernel) to their corresponding Hugging Face parameter paths. This function should handle both scanned and unscanned layer scenarios.
#### Create Hook Function: 
Add a new function (e.g., [MODEL]_MAXTEXT_TO_HF_PARAM_HOOK_FN) that returns a dictionary mapping MaxText parameter names to transformation functions (hooks). These hooks handle value transformations like padding embeddings, rescaling RMSNorm, or transposing/reshaping kernels.
#### Register Mappings: 
At the very bottom of the file, add your new model to the PARAM_MAPPING and HOOK_FNS dictionaries.

### 2. [src/maxtext/checkpoint_conversion/utils/hf_model_configs.py](../../../../checkpoint_conversion/utils/hf_model_configs.py)
This file provides the target Hugging Face architecture configuration that to_maxtext.py and to_huggingface.py rely on.
#### Define Model Config: 
Create the model dictionary/configuration object using Hugging Face's transformers library (e.g., transformers.[ModelName]Config(...)).
#### Register Config: 
At the bottom of the file, add your new model to the HF_MODEL_CONFIGS dictionary, mapping the MaxText model name to the instantiated Hugging Face configuration object.

### 3. [src/maxtext/utils/globals.py](../../../../utils/globals.py)
This file keeps track of official Hugging Face identifiers for the tokenizer.
#### Update HF_IDS: 
Add your model's name to the HF_IDS dictionary, mapping your MaxText model key to the official Hugging Face Hub repository string (e.g., "model_name": "organization/model_repo"). This is necessary for fetching the correct vocabulary and safetensors.index.json.

### Clean up the codebase, no other modification is expected. 

## Final Goal
The final goal of the checkpoint conversion process is to ensure perfect mathematical equivalence between the original Hugging Face model and its newly onboarded MaxText implementation.

Here is how the end-to-end workflow accomplishes this:

[to_maxtext.py](../../../../checkpoint_conversion/to_maxtext.py) (Inward Conversion): You use this script to pull the Hugging Face weights, apply your new transformations (reshaping, padding, norm scaling) from param_mapping.py, and save them out as a MaxText-compatible Orbax checkpoint.
[to_huggingface.py](../../../../checkpoint_conversion/to_huggingface.py) (Outward Conversion): You use this script to ensure the reverse is also perfectly reproducible, transforming the MaxText checkpoint back into the standard Hugging Face format.
[forward_pass_logits_checker.py](../../../../../../tests/utils/forward_pass_logit_checker.py) (Verification): This is the ultimate source of truth. It runs a forward pass on your newly converted MaxText model using a set of text prompts and compares the resulting output logits against a "golden" reference file of logits generated by the original Hugging Face model.
If your mappings and architectural configurations in param_mapping.py and hf_model_configs.py are correct, the checker will pass—meaning the output probability distributions (KL divergence), top-k token rankings, and raw numerical logits match the Hugging Face baseline within a strict tolerance. This guarantees the model was onboarded successfully and will behave exactly as intended during training and inference.