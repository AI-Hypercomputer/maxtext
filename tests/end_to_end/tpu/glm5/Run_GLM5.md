# Run GLM-5.1 on TPU

This directory contains end-to-end integration and benchmark tests for running GLM-5.1 (744B MoE) on Google TPUs.

## Supported Models
* `glm5.1-744b`: 744B total parameters, 75 MoE layers, 256 experts (8 routed experts per token), v_head_dim=256, RoPE interleave=True.

## Workflow Overview

### Step 1: Checkpoint Conversion (`1_test_glm5.sh`)
Runs on CPU/host to convert HuggingFace safetensor checkpoints (`bfloat16`) to MaxText-compatible Orbax checkpoints:
- **Scanned checkpoints:** Optimized for distributed pre-training and fine-tuning.
- **Unscanned checkpoints:** Optimized for high-throughput decoding and inference.

### Step 2: TPU Training & Logit Verification (`2_test_glm5.sh`)
Runs on a 64-chip (`4x4x4`) TPU v5p slice to verify:
1. **Forward Pass Logit Parity:** Validates KL divergence against golden HuggingFace logits (`KL <= 0.3`).
2. **Distributed Pre-Training Benchmark:** Executes multi-host training using:
   - Optimal mesh sharding: `TP=1, EP=4, FSDP=16` (`ici_fsdp_parallelism=-1` automatically divides remaining chips).
   - Splash/Flash Attention tiling: `sa_block_*=512` (configured to fit TPU v5p 16MB VMEM limit for `v_head_dim=256`).
   - Megablox ragged MoE GMM kernels (`megablox=True`, `sparse_matmul=True`).
   - Zero-memory SGD optimizer state (`opt_type=sgd`).
3. **Decoding & Generation:** Validates text generation with `decode.py`.
