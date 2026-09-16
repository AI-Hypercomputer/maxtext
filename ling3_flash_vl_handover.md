# Handover: Ling-3.0-flash-VL Text-Only MaxText Onboarding

## Status: MINI-SUBSET-CKPT FULLY VERIFIED (LOGITS + TOKEN GENERATION)

### 1. Architectural Summary
- Model: `inclusionAI/Ling-3.0-flash-VL` (text-only backbone `bailing_moe_v3`).
- Inhomogeneous 6-layer cycle:
  - Sub-layers 0..4: Kimi Delta Attention (KDA) linear attention with depthwise Causal Conv1D (kernel_size=4), decay gate $g = -5.0 \cdot \sigma(\exp(A_{\text{log}}) \cdot (f(x) + \text{bias}))$, and L2-normalized queries/keys.
  - Sub-layer 5: Multi-head Latent Attention (MLA).
  - Layers 0, 1: Dense SwiGLU MLP (intermediate dim 6144).
  - Layers 2..41: MoE (512 routed experts + 1 shared expert, top-k=8).
- Final RMSNorm & LM Head: Vocab size 157,184, hidden dimension 2,560.

### 2. Artifacts Created & Locations
- **Mini Safetensors Subset (1-layer N=1)**:
  - `/dev/shm/hf_mini/ling3_flash_vl_1layer/model.safetensors` (1,830,061,696 bytes, 21 tensors)
  - Configs & Tokenizer: `config.json`, `text_config.json`, `tokenizer.json`, etc.
- **MaxText Model Definition**:
  - `src/maxtext/models/ling3.py`: `Ling3CausalConv1D`, `Ling3RMSNormGated`, `Ling3KimiDeltaAttention`, `Ling3DenseMlp`, `Ling3DecoderLayer`, `Ling3ScannableBlock`.
  - `src/maxtext/layers/nnx_decoders.py`: Registered `DecoderBlockType.LING3`.
  - `src/maxtext/configs/models/ling3-flash-vl.yml` & `ling3-flash.yml`.
- **Golden Logits Reference**:
  - `/dev/shm/hengtaoguo/golden_mini_logits.jsonl`
  - `/dev/shm/hengtaoguo/golden_mini_logits.json`
- **Converted Orbax Checkpoint**:
  - `/dev/shm/hengtaoguo/checkpoints/ling3_flash_vl_mini_orbax/0/` (Orbax standard format with `_CHECKPOINT_METADATA`, `commit_success.txt`, `items`).

### 3. Quantitative Verification Results
- **Prompt**: `"I love to"` (Tokens: `[40, 2318, 297]`)
- **Forward Pass Logits Parity**:
  - Max Logit Difference: `2.255917e-02`
  - Mean Logit Difference: `3.342941e-03`
  - **KL Divergence**: `9.588060e-06` (Target threshold: $\le 10^{-3}$) -> **PASSED**
- **16-Token Greedy Autoregressive Generation**:
  - MaxText Generated: `[40, 2318, 297, 103250, 2530, 31718, 26572, 119023, 9218, 14239, 33040, 19065, 4665, 6473, 145693, 11333, 78730, 12573, 1866]`
  - HuggingFace Golden: `[40, 2318, 297, 103250, 2530, 31718, 26572, 119023, 9218, 14239, 33040, 19065, 4665, 6473, 145693, 11333, 78730, 12573, 1866]`
  - Decoded Text: `'I love toiret素 organizingisan Ruf拳头的Satellite触诊 fyjavirtuousness'`
  - **Exact Match**: **100% Match across all 19 tokens** -> **PASSED**
