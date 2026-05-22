<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Adding a New Model for LoRA Fine-Tuning

This guide explains how to add Low-Rank Adaptation (LoRA) support for a new model architecture in MaxText.

MaxText leverages [Tunix](https://github.com/google/tunix) and [Qwix](https://github.com/google/qwix) to support Parameter-Efficient Fine-Tuning (PEFT) on JAX/NNX model definitions. Since the architecture uses modular APIs, adding LoRA support for a new model is highly streamlined.

______________________________________________________________________

## 1. Step-by-Step Bring-up Guide for NNX LoRA

To enable LoRA support for a new model, follow these two simple steps:

### Step 1.1: Verify Base Model Support

The target model architecture must already be implemented and supported as a base model in MaxText.

- The JAX/NNX model definition should be located under `src/maxtext/models/` (e.g., \[gemma3.py\](file:///home/jackyf_google_com/maxtext/src/maxtext/models/gemma3.py)).
- The model configurations must be registered and runnable for baseline pre-training or full fine-tuning.

### Step 1.2: Define Trainable LoRA Target Modules

Add a recommended target pattern for your model architecture prefix in \[src/maxtext/configs/post_train/lora_module_path.yml\](file:///home/jackyf_google_com/maxtext/src/maxtext/configs/post_train/lora_module_path.yml):

```yaml
your_model_prefix: "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))"
```

> [!NOTE]
> MaxText's `_get_lora_module_path` in `lora_utils.py` automatically handles both **scanned** (e.g., `layers/0/self_attention/...`) and **unscanned** (e.g., `layers/self_attention/...`) layer formats by injecting an optional layer index regex. You only need to define standard, unscanned paths.

If no prefix matches your model name, MaxText falls back to the `default` pattern:

```yaml
default: "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))"
```

______________________________________________________________________

## 2. Integrating Custom Weight Mappings (When is it needed?)

Determining whether you need to implement custom weight mappings depends entirely on your downstream workflow:

### Scenario A: SFT Training & Conversion to PEFT (No Mapping Needed)

If you only need to run SFT fine-tuning with LoRA and then export the adapter back to Hugging Face format using `to_huggingface.py`, **you do not need to write any custom weight mappings.**

- The conversion utility automatically maps, scales, and formats the LoRA adapter parameters back into standard Hugging Face PEFT format based on the base model's existing weight mapping.

### Scenario B: Decoding with the MaxText vLLM Adapter (Mapping is Required)

If you want to perform decoding or run high-performance serving on your adapted model using the **MaxText vLLM adapter** (e.g., via `vllm_decode`), **you must define and register a custom weight mapping.** This allows the vLLM JAX wrapper to dynamically map and feed weights to the vLLM engine.

To add weight mapping for vLLM decode:

1. **Create a Weight Mapping Config**:
   Create a new file in \[src/maxtext/integration/tunix/weight_mapping/\](file:///home/jackyf_google_com/maxtext/src/maxtext/integration/tunix/weight_mapping/) (e.g., `your_model.py`) defining a mapping dataclass. You can refer to \[gemma3.py\](file:///home/jackyf_google_com/maxtext/src/maxtext/integration/tunix/weight_mapping/gemma3.py) or \[llama3.py\](file:///home/jackyf_google_com/maxtext/src/maxtext/integration/tunix/weight_mapping/llama3.py) as templates.

   Your class should specify:

   - `to_hf_mapping()`: Maps MaxText base parameters to Hugging Face parameters and specifies their sharding axes.
   - `to_hf_hook_fns()`: Custom hook functions for complex weight transformations (e.g., RoPE reordering or query scaling).
   - `lora_to_hf_mappings()`: Custom mapping for LoRA weights if they require different handling.

2. **Register the Mapping**:
   Register your new class in \[src/maxtext/integration/tunix/weight_mapping/__init__.py\](file:///home/jackyf_google_com/maxtext/src/maxtext/integration/tunix/weight_mapping/__init__.py) inside the `StandaloneVllmWeightMapping` class:

   ```python
   # Inside StandaloneVllmWeightMapping
   if name.startswith("your_model_name"):
       return YOUR_MODEL_VLLM_MAPPING
   ```
