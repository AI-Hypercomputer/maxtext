# Mini-Model-Verification & Multi-Token Masked Prefill

This guide covers Phase 4 of the MaxText model bring-up workflow: running [mini-model-verification](mini_verification.md), validating full-model **Multi-Token Masked Prefill** on both `scanned` and `unscanned` checkpoints, and iteratively resolving discrepancies.

## 1. Fast Mini-Model-Verification Workflow

Always verify a representative trimmed/sliced subset ($N$ layers) first using `/dev/shm/hf_mini/{model_name}_xlayers` and `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax` before running full checkpoints. See [mini_verification.md](mini_verification.md) for the full `mini-model-verification` checklist.

---

## 2. Generate Golden Multi-Token Masked Prefill

Use `tests/utils/zero_gen_logits_checker.py` (`tokenize_golden_batch` over `GOLDEN_BATCH_SECTIONS` or evaluation prompts such as TinyMMLU) to build padded multi-token batches with `decoder_segment_ids` and a completion `target_mask`, then save HuggingFace reference `tokens`, `logits`, `target_mask`, and optional `routing_indices` / `indexer_selected_blocks` via `save_golden_npz` (or generate `.jsonl`/`.pkl` goldens via `tests.assets.logits_generation.generate_hf_golden_logits`).

---

## 3. Run Full-Model Multi-Token Masked Prefill (`scanned` & `unscanned`)

Convert the full checkpoint with `to_maxtext` for both `scan_layers=true` (`scanned`) and `scan_layers=false` (`unscanned`), keeping strict `fp32` modules (such as MoE routers/biases and residual gates) in `float32` when saving `bfloat16`. Execute from the MaxText root directory on both checkpoints:

```bash
export HF_HOME=/dev/shm/$USER

# 1. Masked prefill parity: routing/indexer + masked target logits (repeat with scan_layers=false)
python3 -m tests.utils.zero_gen_logits_checker \
  --mode=prefill_logits \
  --golden_tensor_path=/tmp/golden_fp32.npz \
  src/maxtext/configs/base.yml \
  model_name={model_name} \
  load_parameters_path={orbax_ckpt_path} \
  scan_layers=true \
  dtype=float32 weight_dtype=float32 matmul_precision=highest

# 2. Masked CE / TinyMMLU loss delta in nats
python3 -m tests.utils.zero_gen_logits_checker \
  --mode=tinymmlu \
  --golden_tensor_path=/tmp/golden_fp32.npz \
  --tol_nats=0.01 \
  src/maxtext/configs/base.yml \
  model_name={model_name} \
  load_parameters_path={orbax_ckpt_path} \
  scan_layers=true \
  dtype=bfloat16 weight_dtype=bfloat16
```

### Key Evaluation Metrics:
1. **Standard / `fp32` Masked Prefill**: Per-token cosine similarity $\ge 0.9999$, KL divergence $\le 10^{-4}$, max abs error $\le 10^{-2}$ (scaled by $\sqrt{\text{seq\_len}/512}$ on masked tokens), `zero_norm_token_count == 0`, and multi-row vs single-row `mean_batch_diff <= 1e-4`.
2. **Deep `bf16` MoE Models**: Run both:
   - **Router Replay (`forced_routed_experts`)** vs `fp32` golden: min cosine sim $\ge 0.999$, mean KL $\le 0.005$, max abs $\le 1.0$, $\Delta\text{CE} \le 0.01\text{ nats}$.
   - **Unforced End-to-End**: `fp32` matches `fp32` golden; `bf16` achieves mean cosine sim $\ge 0.999$, mean KL $\le 0.005$, and TinyMMLU / masked $\Delta\text{CE} \le 0.01\text{ nats}$.

---

## 4. Training & SFT Correctness Smoke Check (If Applicable)

Verify multi-step training stability (`remat_policy=full`) on both `scanned` and `unscanned` full checkpoints (if TPU HBM is insufficient on your TPU VM or cluster, apply the HBM-saving ladder in [wiki/hbm_verification.md](wiki/hbm_verification.md)):
1. **Pre-training**: Run 5–10 steps of `maxtext.trainers.pre_train.train` on C4 (from converted checkpoint and/or random init) and confirm finite, matching `scanned` / `unscanned` loss trajectories.
2. **SFT**: Run 5–10 steps of `maxtext.trainers.post_train.sft.train_sft_native` with assistant-only completion masking (and optional `--mode=holdout_ce` check).

---

## 5. Iterative Debugging Loop (Closing the Gap)
If KL divergence is high or masked prefill metrics fail:

> [!IMPORTANT]
> **Free Shared Memory Space**:
> If checks fail, immediately delete the converted Orbax checkpoint from `/dev/shm/$USER/checkpoints` using `rm -rf <path_to_checkpoint>` before fixing the bug and running conversion again.

### Step-by-Step Debugging Guide:
1. **Is it off from token 0 / layer 0?**
   - Check token embedder weights, normalization/scaling, transposition, and input ID mapping.
2. **Is it off in attention?**
   - Check if RMSNorm epsilon (and `1 + w` vs standard RMSNorm) matches HF `config.json`.
   - Verify query/key/value projection transpositions, QK-norm/output gating, and head dimension layout.
   - Rotary Embeddings (RoPE): Check if RoPE frequency theta or `partial_rotary_factor` matches, and verify whether `permute_to_match_maxtext_rope` is needed.
   - Check `sliding_window_size` vs global attention layer cycle indexing and `decoder_segment_ids` padding masks.
3. **MLP & MoE routing**:
   - Check expert gate weights, `fp32` router precision, top-k gating probability normalization (`norm_topk_prob`), and compare Router Replay vs Unforced to separate `bf16` router boundary tie-flips from weight/math bugs.

### Deep Verification via Intermediate Activations Mapping:
If discrepancies persist:
1. Run both HF PyTorch and MaxText with $N=1$ layer and feed identical input tokens.
2. Print activations at each sublayer:
   - Input embeddings
   - Pre-attention norm output
   - Attention Q, K, V projections and attention output
   - Post-attention / MLP norm output
   - MLP intermediate and output projections
   - Final norm and logits
3. Compare intermediate tensors side-by-side to pinpoint the exact failing operation.

---

## 6. Final Session Wrap-up & Handover Document
Once `mini-model-verification` passes and the full model checkpoints (`scanned` & `unscanned`) are validated:
1. Finalize `{model_name}_handover.md` in the MaxText root directory using the template in [roadmap.md](roadmap.md).
2. Ensure every completed stage pairs its **quantitative result table**, **exact reproduction command**, and **log/checkpoint/golden artifact paths** alongside the HF architectural findings and resolved pitfalls so all evidence is easy to audit.
