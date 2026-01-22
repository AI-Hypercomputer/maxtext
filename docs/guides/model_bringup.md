<!--
 Copyright 2023–2026 Google LLC

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

# MaxText Model Bringup: Community Contributor Guide

This documentation acts as the primary resource for efficiently integrating new models into the MaxText ecosystem. It provides the community with a standardized workflow covering architectural analysis, feature implementation, and correctness verification.

## 1. Architecture Analysis

The first phase involves determining how the new model's architecture aligns with MaxText's existing capabilities. To facilitate this assessment, refer to the [MaxText architecture overview](https://maxtext.readthedocs.io/en/latest/reference/architecture/architecture_overview.html) and [list of supported models](https://maxtext.readthedocs.io/en/latest/reference/models/supported_models_and_architectures.html).

**Input Data Pipeline**: MaxText supports HuggingFace, Grain, and TFDS pipelines ([details](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline.html)). While synthetic data is typically used for initial performance benchmarks, the framework supports multiple modalities including text and image (audio and video - work in progress).

**Tokenizer**: Supported [tokenizer options](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/input_pipeline/tokenizer.py) include `TikTokenTokenizer`, `SentencePieceTokenizer`, and `HFTokenizer`.

**Self-Attention & RoPE**: Available mechanisms include optimized [Flash Attention](https://github.com/AI-Hypercomputer/maxtext/blob/62ee818144eb037ad3fe85ab8e789cd074776f46/src/MaxText/layers/attention_op.py#L1184) (supporting MHA, GQA, and MQA), Multi-head Latent Attention ([MLA](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/attention_mla.py)), and [Gated Delta Network](https://github.com/AI-Hypercomputer/maxtext/blob/62ee818144eb037ad3fe85ab8e789cd074776f46/src/MaxText/models/qwen3.py#L358). MaxText also supports [Regular](https://github.com/AI-Hypercomputer/maxtext/blob/88d2ffd34c0ace76f836c7ea9c2fe4cd2d271088/MaxText/layers/embeddings.py#L108), [Llama](https://github.com/AI-Hypercomputer/maxtext/blob/88d2ffd34c0ace76f836c7ea9c2fe4cd2d271088/MaxText/layers/embeddings.py#L178), and [YaRN](https://github.com/AI-Hypercomputer/maxtext/blob/88d2ffd34c0ace76f836c7ea9c2fe4cd2d271088/MaxText/layers/embeddings.py#L282) variations of Rotary Positional Embeddings (RoPE).

**Multi-Layer Perceptron (MLP)**: The framework supports both traditional dense models and Mixture of Experts (MoE) architectures, including [configurations](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html) for routed and shared experts.

**Normalization**: We support different [normalization strategies](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/normalizations.py), including RMSNorm and Gated RMSNorm. These can be configured before or after attention/MLP layers.

**Decoder Layers**: Models can have multiple [decoder layers](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/models) with varying structures. The trend has evolved from entirely dense layers to purely MoE layers, and now towards a mix of both.

## 2. (Optional) Feature Implementation

This step can be bypassed if the current MaxText codebase already supports all components required for your model architecture. However, if your model introduces unique logic or requires specific code refactoring, these modifications should be completed before you begin converting checkpoints.

**Sharding**: MaxText supports both auto and explicit sharding modes and provides dedicated sharding functions. We recommend developers use MaxText-specific sharding functions, such as `MaxText.sharding.maybe_shard_with_name`, instead of default JAX sharding hint like `jax.lax.with_sharding_constraint` for better performance.

## 3. Checkpoint Conversion

While most open-source models are distributed in Safetensors or PyTorch formats, MaxText requires conversion to the [Orbax](https://orbax.readthedocs.io/en/latest/) format.

There are [two primary formats](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/checkpoints.html) for Orbax checkpoints within MaxText, and while both are technically compatible with training and inference, we recommend following these performance-optimized guidelines:

- **Scanned Format**: Recommended for **training** as it stacks layers for efficient processing via `jax.lax.scan`. To enable this, set `scan_layers=True`.
- **Unscanned Format**: Recommended for **inference** to simplify loading individual layer parameters. To enable this, set `scan_layers=False`.

### 3.1 Create Mapping

Success starts with a clear map. You must align the parameter names from your source checkpoints (Safetensors/PyTorch) with the corresponding MaxText internal names.

- You can print out the keys and shapes of your original `.safetensors` or `.pth` files.
- To see the target structure, you can initiate a pre-training run to save a randomly initialized checkpoint for inspection.

### 3.2 Write Script

Use existing model scripts within the repository as templates to tailor the conversion logic for your specific architecture. We strongly recommended to use the [checkpoint conversion utility](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/convert_checkpoint.html) rather than [standalone scripts](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/checkpoint_conversion/standalone_scripts).

### 3.3 Verify Compatibility

You can now fine-tune in MaxText using your converted scanned checkpoint, or perform decoding with your unscanned checkpoint. This assumes no compiler or shape errors are encountered.

## 4. Unit Tests

Before verifying the entire model, it is highly recommended to perform unit tests on new layers or features. This approach helps isolate potential bugs early by comparing the MaxText JAX implementation against a reference PyTorch implementation—typically from HuggingFace or the original model release. More examples can be found [here](https://github.com/search?q=repo%3AAI-Hypercomputer%2Fmaxtext+vs_reference&type=code).

Core Strategy:

- **Instantiate Layers**: Create instances of both the MaxText JAX layer and the corresponding PyTorch layer.
- **Copy Weights**: Initialize the PyTorch layer and copy its exact weights to the JAX layer instance. This ensures both start from the same state. Pay close attention to weight naming and potential shape differences (e.g., transposing Linear/MLP weights).
- **Prepare Input**: Create identical input data (e.g., random tensors) for both frameworks. Use a helper to convert PyTorch tensors to JAX arrays.
- **Forward Pass**: Run the input through both layers. Remember to set the PyTorch model to evaluation mode (`model_pt.eval()`) to disable dropout etc.
- **Compare Outputs**: Convert the PyTorch output to a JAX array (or NumPy array) and use `numpy.testing.assert_allclose` to check if the outputs are numerically close within a specified tolerance (atol, rtol).

## 5. End-to-end correctness

This verification process can vary in duration. If you're working with a small model, you're fortunate as it allows for rapid iteration on your development machine. To verify a model's correctness, we could leverage two strategies below - comparing logits and evaluation.

### 5.1 Compare Forward Logits

This is the primary verification for training, using a small set of input prompts. Typically, we obtain logits from both a reference implementation and MaxText, then compare their divergence to assess consistency.

When running the comparison script, using the flags `weight_dtype=float32 dtype=float32 activations_in_float32=true matmul_precision=float32 float32_logits=true float32_qk_product=true` should result in a smaller divergence. Ideally, this will meet the criteria of `--max_kl_div=1e-4` or combination of `--atol=1e-02` and `--rtol=1e-02`.

For models with existing Hugging Face support, you can validate parity using the following methods:

- **Real-time Comparison**: Directly run this [forward_pass_logit_checker.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/utils/forward_pass_logit_checker.py) script with the `--run_hf_model=True` and `--hf_model_path=<HF-model-name>` flags to perform an on-the-fly logit comparison.
- **Golden Logit Validation**: Compare your model against pre-saved reference logits ([script](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/assets/logits_generation/generate_hf_golden_logits.py)) by providing a path via `--golden_logits_path`. Supported formats include JSON Lines (`.jsonl`) and pickle (`.pickle` or `.pkl`).

### 5.2 Eval Benchmark

MaxText integrates with benchmark libraries like [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [evalchemy](https://github.com/mlfoundations/evalchemy) to facilitate rapid verification of common inference scores ([guide](https://github.com/AI-Hypercomputer/maxtext/tree/main/benchmarks/api_server)). This is particularly useful for validating decoding outputs or assessing model performance when logits deviate slightly from reference values.

## 6. Completion Checklist

Please ensure all items on the following checklist are completed before finalizing your MaxText model bring-up support:

1. Core Functionality

- [ ] Implement all new required features for functionality, and have unit tests against the reference implementation.
- [ ] Update FLOP calculations if new components are added.

2. Checkpoint Conversion

- [ ] Complete the checkpoint conversion from source to Orbax for both: scanned version & unscanned version.

3. Model & Integration Verification

- [ ] Verify model forward logits using common prompts from the repository: scanned version & unscanned version.
- [ ] Perform common downstream benchmark scores (e.g., MMLU) against the reference.

4. Documentation

- [ ] Create a user guide and post an announcement in the MaxText repo.

## Community Q&A (FAQ)

**Q: How do I debug code inside a JAX JIT function?**

**A:** Standard debuggers like VSCode cannot step through `jitted` functions. Use `jax.debug.print("{item}", item=...)` to inspect values during execution.

**Q: How to debug the correctness of logits?**
If you run the `forward_pass_logit_checker.py` to compare reference logits with your implementation and find divergence, we highly recommend checking the logits in the first layer using the same prompt. A few things to start with:

- Weight loading logic: The sequence of operations during weight loading is critical; for instance, you may need to split or reshape components before you concatenate them to ensure the final matrix matches the reference structure.
- Naming conflicts: Ensure that internal variables and functions have unique names; simple naming collisions can cause issues to correctly pass RoPE or QK-norm intervals to the attention mechanism.
- Query scaling: To avoid scaling issues, query scaling was moved from the checkpoint conversion phase to the forward pass using the `query_pre_attn_scalar` argument.

**Q: How to compile models for a target hardware without physical access?**

**A:** If you need to compile your training run ahead of time, use the train_compile.py tool. This utility allows you to compile the primary train_step for specific target hardware without needing the actual devices on hand. It’s particularly useful for verifying your implementation's functionality on a local Cloud VM or a standard CPU. Please refer [here](https://maxtext.readthedocs.io/en/latest/guides/monitoring_and_debugging/features_and_diagnostics.html#ahead-of-time-compilation-aot) for more examples.

**Q: My model is too large for my development machine. What should I do?**

**A:** You can create a smaller version of the model (fewer layers or smaller hidden dimensions) to iterate on your local box before moving to a larger cluster.

**Q: How to store logits from a JAX JIT function?**

We generally use jax.debug.print() to print out results; however, those values are often truncated, so storing them directly is recommended to compare the final logits, especially running on GKE clusters.

```
def save_with_jit(x):
  jnp.save("your_file_name.npy", x)
jax.debug.callback(save_with_jit, your_variable)
```

**Q: Do I need to create a scanned checkpoint before an unscanned one?**

**A:** No, you can directly generate an unscanned checkpoint from the source weights using the conversion utility.
