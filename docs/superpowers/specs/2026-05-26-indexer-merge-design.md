# DeepSeek Indexer Merge Design

## Overview
This document outlines the architectural design for merging the training logic of the DeepSeek-V3 `Indexer` (found in `src/maxtext/layers/attention_mla.py`) and the DeepSeek-V4 `DeepSeekV4Indexer` (found in `src/maxtext/layers/attention_compressed.py`). Both indexers share a common mathematical core for scoring relevance (ReLU similarity, dynamic head weighting, and Top-K selection) but differ significantly in how they construct their key representations and apply causal masking.

The goal is to eliminate code redundancy while maintaining the strict architectural boundaries required for token-level (V3) and block-compressed (V4) attention paradigms.

## Architecture: Base Class Inheritance
The chosen approach utilizes a Base Class Inheritance model. This provides excellent code reuse for the shared scoring kernel and query projection logic, while allowing the subclasses to independently manage their mutually exclusive key preparation and masking strategies.

### 1. `BaseIndexer` (The Core Kernel)
A new `BaseIndexer(nnx.Module)` will be created to house the shared logic.

**Responsibilities:**
*   **Parameter Ownership:** It will initialize and own the shared projection parameters:
    *   `q_b_proj` (or `wq_b`): The dense layer that projects the low-rank latent query representation into the multi-head query space.
    *   `weights_proj`: The dense layer that projects the inputs to determine dynamic head importance scaling.
    *   `softmax_scale` & `weights_scaling`: The scaling constants applied for numerical stability.
*   **Query Preparation (`prepare_query`):** A method that takes the low-rank query tensor, applies the `q_b_proj`, reshapes it to the multi-head layout, and applies Rotary Position Embeddings (RoPE).
*   **Scoring & Selection (`compute_topk`):** A method that takes the prepared `Q`, the prepared `K`, the un-projected `inputs_q`, and a resolved `mask`. It executes the core scoring formula:
    1.  `Logits = ReLU(Q @ K.T)`
    2.  `Head_Weights = weights_proj(inputs_q) * scale`
    3.  `Score = (Logits @ Head_Weights) + mask`
    4.  `Indices = jax.lax.top_k(Score, k)`

### 2. `MLAIndexer` (DeepSeek-V3 Token-Level)
The existing `Indexer` in `attention_mla.py` will be renamed/refactored to inherit from `BaseIndexer`.

**Responsibilities:**
*   **Key Preparation:** Projects tokens directly into `K` using a standard `wk` projection, followed by RMS normalization and RoPE.
*   **Masking:** Resolves standard token-level 2D/3D causal attention masks (and padding masks if using cached values).
*   **Execution Flow:** Calls `self.prepare_query()`, prepares its token-level `K`, constructs its token-level masks, and calls `self.compute_topk(Q, K, mask)`.

### 3. `DeepSeekV4Indexer` (DeepSeek-V4 Block-Compressed)
The `DeepSeekV4Indexer` in `attention_compressed.py` will be refactored to inherit from `BaseIndexer`.

**Responsibilities:**
*   **Key Compression:** Groups tokens into overlapping sliding windows (`Ca`/`Cb` chunks of size `compress_rate`). It computes a Softmax-weighted pooling using `gate_proj` and `position_bias`, normalizes the pooled representation, and applies RoPE to yield `K_pooled`.
*   **Structural Masking:** Computes the V4-specific block mask to prevent queries from attending to compressed blocks that represent future tokens (`causal_threshold = (position_ids + 1) // compress_rate`).
*   **Execution Flow:** Calls `self.prepare_query()`, prepares its compressed `K_pooled`, constructs its block-level future mask, and calls `self.compute_topk(Q, K_pooled, block_mask)`.

## Data Flow Summary
```
[Inputs] -> Subclass
   |
   +-> Subclass delegates `low_rank_q` to BaseIndexer.prepare_query() -> [Q]
   |
   +-> Subclass prepares `K` (Token-level for V3, Block-pooled for V4) -> [K]
   |
   +-> Subclass prepares `mask` (Token causal for V3, Block causal for V4) -> [Mask]
   |
   +-> Subclass passes Q, K, Mask to BaseIndexer.compute_topk() -> [Top-K Indices]
```

## Testing & Validation
*   Existing unit tests for `attention_mla.py` and `attention_compressed.py` must pass without modification to their expected mathematical outputs, ensuring the refactor is bit-accurate.
*   No main-loss gradients should flow into the `MLAIndexer` during training, preserving the gradient isolation strategy utilizing `jax.lax.stop_gradient`.