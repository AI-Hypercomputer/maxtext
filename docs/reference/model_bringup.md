<!-- # New Model Bringup Guide

**Description:**
This PR adds a new guide, `docs/model_bringup.md`, detailing the standard process for onboarding new model architectures into MaxText.

This guide covers:

1. Architecture Analysis
2. Feature Implementation
3. Checkpoint Conversion (PyTorch to Orbax)
4. Verification & Testing
5. Completion Checklist

This addresses the need for a standardized workflow for external contributors adding support for new models (e.g., DeepSeek, Llama variations, etc.).
 -->


# MaxText New Model Bringup Guide

This guide outlines the step-by-step process for bringing up a new model architecture in MaxText. It is designed for developers who want to add support for models that are not yet available in the repository.

## 1. Analyze Architecture & Gap Analysis

Before writing code, compare the new model's architecture against existing MaxText capabilities. Most modern LLMs share similar components, but subtle differences in normalization, attention, or position embeddings can prevent convergence.

### Key Components to Review:

* **Input Pipeline:** Does the model use a standard tokenizer (e.g., TikToken, SentencePiece, HFTokenizer)?
* **Attention Mechanism:**
* Does it use standard Dot-Product Attention or variants like Multi-Head Latent Attention (MLA)?
* Check the RoPE (Rotary Positional Embedding) implementation. Are there unique scaling factors or frequency calculations?


* **Normalization:** Check for RMSNorm, LayerNorm, or QK-Norm.
* **Activation Functions:** Identify if the model uses SwiGLU, GELU, or a custom activation.
* **Decoder Layer Structure:** Verify the ordering of Attention, MLP, and Normalization layers.

## 2. Implement Missing Features

If the gap analysis reveals unsupported features, implement them in the `maxtext/layers/` directory.

* **Refactoring:** If an existing layer is *almost* correct but needs a slight modification (e.g., a new argument for RoPE), prefer refactoring the existing layer over duplicating code.
* **New Layers:** For distinct architectural changes (e.g., a specific MoE routing logic or a new attention variant like MLA), create a new class in the appropriate layers file.

## 3. Checkpoint Conversion

MaxText uses **Orbax** for checkpointing. To run inference or fine-tuning, you must convert the original model weights (usually from HuggingFace/PyTorch) into the MaxText Orbax format.

### Conversion Strategy

You will need to write a conversion script that maps the variable names from the source model to the MaxText parameter tree.

1. **Load Source Weights:** Load the PyTorch state dictionary.
2. **Initialize MaxText Model:** Instantiate the MaxText model with random weights to generate the target PyTorch tree structure (using `jax.tree_util`).
3. **Map Tensors:** Create a mapping dictionary that links PyTorch layer names (e.g., `layers.0.self_attn.q_proj`) to MaxText layer names (e.g., `layers_0/self_attention/query`).
4. **Permute Weights:** Ensure weight shapes match. JAX often uses `(features_in, features_out)` for kernels, whereas PyTorch may use `(features_out, features_in)`. Transpose where necessary.

> **Note on Scanned vs. Unscanned Checkpoints:**
> * **Scanned Checkpoints:** Recommended for **training**. Weights for identical layers are stacked (e.g., shape `[num_layers, ...]` rather than separate variables like `layer_0`, `layer_1`).
> * **Unscanned Checkpoints:** Recommended for **inference**. Layers are stored as separate variables in the checkpoint tree.
> 
> 

## 4. Verification & Testing

Do not attempt full training until you have verified the correctness of your implementation.

### A. Unit Tests (Layer Level)

Create unit tests for individual components (Attention, MLP, RoPE).

1. Initialize the layer in both PyTorch (reference) and MaxText (JAX).
2. Copy weights from PyTorch to JAX.
3. Feed the same fixed input tensor to both.
4. Assert that the outputs match within a reasonable tolerance (e.g., `atol=1e-5`).

### B. Logit Comparison (End-to-End)

Verify the full forward pass.

1. Load the converted checkpoint into MaxText.
2. Run a forward pass on a specific prompt (e.g., "The quick brown fox").
3. Run the same prompt through the reference HuggingFace model.
4. Compare the output logits. They should match closely.

## 5. Model Bringup Checklist

Use this checklist to ensure the model is fully supported.

* [ ] **Architecture Analysis:** Identified all differences between the new model and existing MaxText layers.
* [ ] **Feature Implementation:** Implemented missing layers (Attention, Norms, etc.) in `maxtext/layers/`.
* [ ] **Checkpoint Conversion:** Created a script to convert weights from HuggingFace to Orbax.
* [ ] **Unit Testing:** Verified critical layers (Attention, MLP) against a PyTorch reference.
* [ ] **Logit Verification:** Confirmed that MaxText forward pass logits match the reference model.
* [ ] **Decode Test:** Verified that the model generates coherent text during inference.
* [ ] **Documentation:** Added instructions on how to run the new model config.



<!-- ### **Next Steps for You**

Would you like me to refine the "Checkpoint Conversion" section to include a code snippet example for mapping tensors, or is the high-level description sufficient for this PR? -->