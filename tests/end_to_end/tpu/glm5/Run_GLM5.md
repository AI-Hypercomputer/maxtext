# Run GLM-5.1 and GLM-5.2 on TPU

This directory contains end-to-end integration and benchmark tests for running GLM-5.1 and GLM-5.2 (744B MoE with Cross-Layer IndexShare) on Google TPUs.

## Supported Models
* `glm5.1-744b`: 744B total parameters, 75 MoE layers, 256 experts (8 routed experts per token), v_head_dim=256, RoPE interleave=True.
* `glm5.2-744b`: 744B total parameters, 75 MoE layers, 256 routed experts + 1 shared expert, Cross-Layer IndexShare (`FSSS` periodic pattern), DSA Sparse Attention with Top-K indexer routing.

## Workflow Overview

### Step 1: Checkpoint Conversion (`1_test_glm5.sh`)
Runs on CPU/host to convert HuggingFace safetensor checkpoints (`bfloat16`) to MaxText-compatible Orbax checkpoints:
- **Scanned checkpoints:** Optimized for distributed pre-training and fine-tuning.
- **Unscanned checkpoints:** Optimized for high-throughput decoding and inference.

### Step 2: TPU Training & Logit Verification (`2_test_glm5.sh`)
Runs on TPU slices (e.g. 64 cores) to verify:
1. **Forward Pass Logit Parity / Generation:** Validates KL divergence against golden HuggingFace logits (`KL <= 0.3`) and text completion.
2. **Distributed Pre-Training Benchmark:** Executes multi-host training using:
   - Optimal mesh sharding: `TP=1, EP=4, FSDP=16`.
   - Cross-Layer IndexShare (`use_index_share=true`, `index_share_pattern="FSSS"`, `prune_shared_indexers=true`).
   - Zero-memory SGD optimizer state (`opt_type=sgd`).
3. **Decoding & Generation:** Validates text generation with `decode.py`.
