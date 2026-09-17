# facebook/DiT-XL-2-256 Bring-up Handover Document

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: `facebook/DiT-XL-2-256`
- **Decoder Pattern**: Diffusion Transformer (DiT) with Adaptive Layer Norm (adaLN-Zero).
- **Key Parameters**:
  - `patch_size`: 2
  - `in_channels`: 4
  - `hidden_size`: 1152
  - `num_attention_heads`: 16
  - `depth`: 28 (Number of DiT Blocks)
  - `vocab_size`: 1000 (Class labels + 1 for CFG)

## 2. Phase-by-Phase Status
- [x] Phase 1: HF Discovery & Configuration (DONE)
- [x] Phase 2: JAX Layer Implementation (DONE - Implemented in `src/maxtext/layers/vae.py`, `src/maxtext/layers/autoencoder_kl.py`, `src/maxtext/models/dit.py`)
- [x] Phase 3: Checkpoint Mapping & Conversion (DONE - Using bundled Orbax checkpoint)
- [ ] Phase 4: E2E Logits & Decode Verification (IN PROGRESS - Images generated but are garbled noise)

## 3. Bugs Encountered & Solutions

### TPU Initialization Mismatch
- **Symptoms**: `GetChip(i)->location().index_on_host() == i` failed or topology mismatch errors when setting subsets like `TPU_VISIBLE_CHIPS=0,1`.
- **Root Cause**: TPU topology and JAX distributed initialization expect all chips to be available unless carefully partitioned.
- **Solution**: Use only TPU chips 4,5,6,7 by setting `TPU_VISIBLE_CHIPS=4,5,6,7`.

### Latent Explosion (Divergence)
- **Symptoms**: Latent stats exploded to huge numbers over diffusion steps.
- **Root Cause**: `DDIMScheduler` default initialization used a `linear` beta schedule, whereas DiT expects `scaled_linear`.
- **Solution**: Updated `sampler_v2.py` to use `scaled_linear` beta schedule with explicit LDM parameters (`beta_start=0.00085`, `beta_end=0.012`).

### High Frequency Noise in Output
- **Symptoms**: Generated images are multicolor noise, despite stable latent stats.
- **Root Cause**: Under investigation. Suspected missing activation function in VAE Decoder's final conv layer or latent scaling/layout mismatch during decoding pipeline.
- **Current hypothesis**: Missing `jnp.tanh` in `src/maxtext/layers/vae.py` on `conv_out`.

## 4. Pitfalls & Model-Specific Gotchas (Session Wrap-up)

### Parameter Flattening Layout
- The `patch_embed` weights are flattened from PyTorch `Conv2d` `(Out, In, H, W)` to JAX `DenseGeneral` kernel. The layout assumed by model `Sequential` is derived from standard row-major traversal of PyTorch weight, meaning inputs must be flattened in corresponding linear order `(C, P_h, P_w)`.
- Internal logic handles this conversion in `sampler_v2.py`.

## 5. Full Executable Reproduction Commands

### Remote Decoding Parallelism 4 (All Chips)
```bash
sh decode_v2_all_chips.sh gs://ly-maxtext/output/dit-xl-2-256/09100945/0/items
```

### Image Validation
```bash
/home/liyinn_google_com/anaconda3/envs/maxtext/bin/python scratch/validate_images.py white_shark_v2.png umbrella_v2.png
```

### Image Recognition Verification
```bash
/home/liyinn_google_com/anaconda3/envs/maxtext/bin/python scratch/verify_image.py white_shark_v2.png
/home/liyinn_google_com/anaconda3/envs/maxtext/bin/python scratch/verify_image.py umbrella_v2.png
```
