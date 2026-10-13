# Mini-Model Verification, E2E Logits & Decode Verification

This guide covers Phase 4 of the MaxText model bring-up workflow: verifying a representative trimmed (mini) version of the target model across **forward boundaries, enabled heads/objectives, backward gradients, and multi-step stateful optimization**, alongside **E2E forward-pass logits validation** and **16-token autoregressive decode checks** before full-scale model verification.

> [!NOTE]
> This guide is model-agnostic. Apply each check to **every distinct sub-layer type, head, and objective your model actually has**, and skip items marked *(if …)* that do not apply (record them as `N/A` in the handover doc).

---

## 1. Fast Mini-Subset Verification Workflow

Always verify a **representative mini model** first: a small configuration of the target model that keeps every distinct component, with **identical weights** on the HuggingFace and MaxText sides. All commands below use these placeholders:
- HF mini model: `/dev/shm/hf_mini/{model_name}_mini`
- `$MINI_CKPT`: the MaxText `load_parameters_path` of the mini checkpoint (e.g. `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax/0/items`)
- `$MINI_SCAN`: `true`/`false`, matching how `$MINI_CKPT` was written (every MaxText command passes `scan_layers=$MINI_SCAN`)
- `$MINI_OVERRIDES`: the MaxText config overrides that describe the mini model, e.g. `override_model_config=true base_num_decoder_layers=4 num_experts=16`

### 1.1 Gate 0: Build the Mini Model (Agent's Choice) & Prove Weight Parity

MaxText has no dedicated mini-model tool. **Choose the approach that fits the model and environment**, write any helper as a throwaway script outside the PR (see [pr_cleanup.md](pr_cleanup.md)), and record the choice, the reason, and the exact commands in `{model_name}_handover.md`.

**1. Decide what the mini model must contain.** Keep one complete architectural cycle of decoder layers (e.g. a full local/global or linear/full attention pattern, any dense prefix layers plus at least one MoE layer), plus every multimodal encoder / projector and auxiliary head (e.g. MTP). Shrink only what is repeated or wide.

**2. Pick a construction approach.** Common options:

| Approach | How | Good when | Watch out for |
|---|---|---|---|
| **A. Slice real HF weights → `to_maxtext`** | Copy `config.json` with fewer layers (and fewer experts if needed), keep only the matching tensors from the safetensors shards (see [hf_analysis.md §4](hf_analysis.md)), then convert with `to_maxtext.py`. | Real weights are accessible and the kept tensors fit in `/dev/shm`. Default choice: real weights also give meaningful decode output. | Expert slicing must slice the router and every expert tensor consistently; tied embeddings; per-layer config lists (e.g. `layer_types`) must be truncated too. |
| **B. Random-init a small HF config → `to_maxtext`** | Build the HF model from a reduced config (`AutoModelForCausalLM.from_config(...)`), re-randomize every parameter, `save_pretrained`, then convert. | Weights are gated, huge, or not yet released; or widths/experts must shrink to fit memory. | Default inits hide bugs (norm scales of 1, zero biases, zero-init gates): **re-randomize all parameters**, including norms, biases, and gate/scale vectors, with non-trivial values. |
| **C. Start from an existing MaxText checkpoint → `to_huggingface`** | Prefix-slice a full MaxText checkpoint to the mini config in a short script, then export with `to_huggingface.py --override_model_architecture`. | A full MaxText checkpoint already exists and the reverse mapping is implemented. | Scanned vs. unscanned layout must match `$MINI_SCAN`; inhomogeneous scanned blocks stack per cycle position; `to_huggingface.py` needs `HF_AUTH_TOKEN`. |

Any other approach is fine if it satisfies the requirements below.

**3. Requirements (regardless of approach):**
- **Representative**: every distinct layer type, head, and objective of the full model is present at least once.
- **Consistent configs**: the HF `config.json` and `$MINI_OVERRIDES` describe the same model (layers, experts, widths, per-layer pattern lists).
- **Non-degenerate weights**: no all-zero / all-one parameters that could mask mapping, offset, or transpose bugs.
- **Gate 0 weight parity**: load both mini checkpoints in the **same dtype** (e.g. `float32`; note `to_maxtext.py` defaults to `--save_dtype=bfloat16`, so pass `--save_dtype=float32` when converting `float32` weights) and verify that every parameter tensor matches with **`max_abs_diff == 0.0`** after applying the param mapping. Any dtype cast during conversion invalidates an exact-zero check.

### 1.2 Strategy First: Architecture-Specific Dimension Analysis & Attribution Principle
Before executing tests, define a verification strategy tailored to the target model's architecture:
- **Identify the Architectural Dimensions That Matter**:
  - *Attention Variants* (e.g., full vs. sliding-window, hybrid linear/recurrent + full attention, gated attention outputs)
  - *Dense vs. Mixture-of-Experts (MoE)* (e.g., dense SwiGLU MLP vs. routed + shared experts)
  - *Multimodal Encoders & Positional Embeddings* (e.g., vision/audio encoder + projector, multi-axis RoPE)
  - *Auxiliary Heads & Objectives* (e.g., Multi-Token Prediction, Z-loss, MoE load-balancing loss)
- **Attribution Principle (Bringup Bug vs. Underlying Shared Behavior)**:
  - Whenever a discrepancy appears on challenging inputs (such as left-padding or packed segments in recurrent/conv layers, or multi-frame inputs in a vision encoder), write a **targeted attribution probe** (e.g., isolating a single layer or injecting a 1-line mask hook) to determine whether the discrepancy was introduced by the new model bring-up or is an underlying behavior of a shared MaxText module.

### 1.3 Numerical Settings & Pass Thresholds
Run core checks in `float32` on both sides: MaxText with `dtype=float32 weight_dtype=float32 matmul_precision=highest`, and the PyTorch reference in `float32` on CPU (or GPU with TF32 disabled). Default thresholds (tighten or relax per model, but record any deviation in the handover doc):

| Check | Metric | Threshold |
|---|---|---|
| Gate 0 weight parity | `max_abs_diff` | `== 0.0` (same dtype) |
| Forward boundaries & logits (Dims 1–3) | `cosine_sim` / `top1_agreement` | `>= 0.999999` / `100%` |
| Loss components (Dim 4) | `abs_diff` | `<= 1e-5` |
| Gradients & VJP probes (Dim 5) | `cosine_sim` | `>= 0.999999` |
| 3-step AdamW weights & moments (Dim 6) | `cosine_sim` | `>= 0.9999` |
| Frozen parameters (Dim 6) | `max_abs_diff` | `== 0.0` (bitwise) |
| `scan_layers` / sharding parity (S.1, S.3) | `cosine_sim` + `max_abs_diff` | `>= 0.999999` and `<= 1e-5` (reduction order differs; do not require exactly `1.0`) |
| `bfloat16` sanity (S.4) | `cosine_sim` / `top1_agreement` | `> 0.998` / `100%`, zero `NaN`/`Inf` |
| CLI logits checker | avg / max KL | `<= 1e-3` / `<= 1e-2` |

---

## 2. The 6 Core Mini-Verification Dimensions

The mini PyTorch/HuggingFace model and the mini MaxText/JAX model must behave identically (within the thresholds in §1.3) across all **6 verification dimensions**:

### Dimension 1: Design Challenging Inputs Deliberately
Test synthetic inputs specifically crafted to stress boundary conditions rather than only simple single-sentence prompts:
- **Test sequence and block/chunk boundaries**:
  - Test sequence lengths straddling every model-specific boundary: causal `conv1d` kernel sizes ($K$), recurrent/linear-attention chunk sizes ($C$), sliding-window sizes, etc. e.g. $S \in \{K-1, K, K+1, C-1, C, C+1\}$ (such as $S \in \{3, 4, 5, 63, 64, 65\}$ for $K=4, C=64$).
  - Verify that partial chunks and single-token remainder chunks (`65 = 64 + 1`) match.
- **Test heterogeneous batches**:
  - Construct batches with variable valid lengths under unpadded, right-padded, and left-padded layouts (`decoder_segment_ids != 0`).
  - If recurrent/conv layers do not consume `decoder_segment_ids`, use a pre-`conv1d` mask attribution probe to verify exact parity.
- **Test packed-document boundaries**:
  - Construct multi-document packed sequences (`decoder_segment_ids = [1..1, 2..2]`) with per-segment resetting `decoder_positions = [0..L_1-1, 0..L_2-1]`.
  - Verify both isolated attention-block parity under block-diagonal causal masks and zero cross-document leakage when Document 2 tokens are perturbed.
- **Test different chat templates and special tokens**:
  - Include the model's chat template framing, BOS/EOS, multimodal delimiter/placeholder tokens, any reasoning or tool-call tokens, and vocabulary boundary token IDs (`0` and `vocab_size - 1`).
- **Test multimodal & positional-embedding inputs** *(if VLM/Omni)*:
  - Feed raw synthetic visual patches (`pixel_values` / `encoder_images`) with explicit position IDs (including multi-axis positions if the model uses them).

### Dimension 2: Check Forward Outputs and Masks at Meaningful Boundaries
Compare **both component-level intermediate outputs and complete small-model logits** (`cosine_sim`, `rel_l2`, `max_abs_diff`, and `top1_agreement`). For each distinct sub-layer type in your model, pick boundaries such as:
1. **Embedding & Multimodal Scatter Boundaries**:
   - Token embeddings (`token_embedder` output) — catches embedding transposition / input-ID mapping bugs.
   - *(if multimodal)* Encoder patch/input embeddings + position embeddings.
   - *(if multimodal)* Encoder block stack + projector output.
   - *(if multimodal)* Post-scatter multimodal input embeddings (`merge_mm_embeddings` in `src/maxtext/multimodal/utils.py`, placing encoder features into multimodal placeholder-token slots — catches silent special-token ID mismatches before the decoder).
2. **Linear / Recurrent Attention Sub-Stages** *(if hybrid/recurrent)*:
   - Input norm $\rightarrow$ fused/split input projections $\rightarrow$ (optional) causal `conv1d` + activation $\rightarrow$ recurrence / state update (including gates) $\rightarrow$ output norm / gating $\rightarrow$ output projection.
3. **Full-Attention Sub-Stages**:
   - Q/K/V projections $\rightarrow$ (optional) `q_norm` / `k_norm` $\rightarrow$ RoPE rotation $\rightarrow$ causal / sliding-window / segment mask application $\rightarrow$ (optional) output gate $\rightarrow$ `o_proj`.
4. **MLP / MoE Sub-Stages**:
   - Post-attention norm $\rightarrow$ gate/up projections (`wi_0`, `wi_1`) & activation $\rightarrow$ down projection (`wo`); for MoE also router logits, top-k selection, and combine weights.
5. **Final Norm, Main LM Head, and Auxiliary Heads**:
   - Final decoder norm $\rightarrow$ `logits_dense` (`lm_head`).
   - *(if MTP)* Target shifting/masking $\rightarrow$ MTP input norms & projection $\rightarrow$ MTP transformer block $\rightarrow$ MTP final norm $\rightarrow$ MTP logits.

### Dimension 3: Give MoE / Dense Its Own Coverage Requirements
- **If the model is MoE**:
  - **Cover routing patterns deliberately**: Test uniform routing, skewed/collapsed routing (all tokens to 1 expert), capacity-boundary routing, and padded-token routing.
  - **Use forced routing if needed**: Override router logits deterministically to isolate expert weight permutation (`wi_0`, `wi_1`, `wo`), `norm_topk_prob`, shared-expert gating, and `moe_lb_loss`.
- **If the model is Dense**:
  - Verify dense SwiGLU/GeGLU intermediate activations (`act(wi_0(x)) * wi_1(x)`) and output projection (`wo`) at the sub-module boundary.
  - Confirm the config is truly dense (`num_experts == 1`, the MaxText default for dense models) and that `aux["moe_lb_loss"] == 0.0` (zero spurious routing overhead).

### Dimension 4: Verify Every Enabled Head and the Actual Training Objective
- **Inspect all enabled outputs**: Main LM head logits (`[B, S, V]`) and, if present, MTP head logits (`[B, S, V]`) and multimodal projector outputs (`[N_merged, D]`).
- **Expose loss components before aggregation**: Inspect the `aux` dict returned by `loss_fn` in `src/maxtext/trainers/pre_train/train.py` — `xent_sum`, `total_weights` (main CE = `xent_sum / total_weights`), `z_loss`, `mtp_loss`, `moe_lb_loss`, and `indexer_loss` (if applicable) — alongside the total loss.
- **Test objectives separately and together** across an ablation matrix (drop rows for objectives the model does not have):
  1. `Config_A` (Main Cross-Entropy Only): `mtp_loss_scaling_factor=0.0, z_loss_multiplier=0.0`
  2. `Config_B` (Main Cross-Entropy + Z-Loss): `mtp_loss_scaling_factor=0.0, z_loss_multiplier=1e-4`
  3. `Config_C` (MTP Objective Isolated) *(if MTP)*: `mtp_loss_scaling_factor=1.0, z_loss_multiplier=0.0` (compare `aux["mtp_loss"]` directly against PyTorch MTP cross-entropy)
  4. `Config_D` (Combined Production Objective): production values (e.g. `mtp_loss_scaling_factor=0.1, z_loss_multiplier=1e-4`), with multimodal inputs if applicable.

### Dimension 5: Make Backward Checks Sensitive and Independent
- **Compare gradients per parameter group and input**:
  - Run `jax.value_and_grad` on the combined objective (`Config_D`) and compare against PyTorch `.backward()` across every parameter group from input to output (e.g., `token_embedder`, each distinct decoder block type and its sub-modules, `decoder_norm`, `logits_dense`, and any vision / MTP parameters).
- **Add block-level upstream-gradient (VJP) probes**:
  - Generate a deterministic synthetic upstream cotangent $G \sim \mathcal{N}(0, 1)$ (fixed seed, e.g. `seed=12345`) and compute the isolated Vector-Jacobian Product $\langle f_{\text{block}}(x, W), G \rangle$ for each distinct block type (e.g., each attention variant, MoE vs. dense block, MTP block).
  - Verify that both the input activation gradient ($\partial L / \partial x_{\text{in}}$) and local parameter gradients ($\partial L / \partial W_{\text{block}}$) match PyTorch completely independently of downstream layers or loss scaling.

### Dimension 6: Test Actual Multi-Step Updates and State Propagation
- **Run several real optimizer updates with changing batches**:
  - Initialize an identical unclipped stateful AdamW optimizer (`optax.inject_hyperparams(optax.adamw)(learning_rate=1e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01)` vs. `torch.optim.AdamW` with matching hyperparameters) and run **3 consecutive training steps** on **3 distinct changing batches**.
- **Compare state at the end of the complete step**:
  - Compare the per-step loss trajectory ($L^{(1)}, L^{(2)}, L^{(3)}$), the updated weights ($W^{(3)}$), and the first and second optimizer moments ($m^{(3)}$ / `mu` and $v^{(3)}$ / `nu`) across all subsystems.
- **Verify trainable/frozen parameter selection**:
  - Test config flags such as `freeze_vision_encoder_params=True` *(if multimodal)* and any masked optimizer parameter selection: assert `max |grad(frozen)| == 0.0`, `max |grad(trainable)| > 0.0`, and exact bitwise weight invariance ($W_{\text{frozen}}^{(3)} \equiv W_{\text{frozen}}^{(0)}$) after 3 optimizer steps.

---

## 3. Supplemental Production-Readiness Checks (Bridging Mini Model to Full Scale)

Before concluding mini-model verification, run these fast supplemental checks to ensure production flags do not introduce regressions:
1. **S.1 `scan_layers=True` vs. `scan_layers=False` Parity** *(always)*:
   - Verify `PARAM_MAPPING(..., scan_layers=True)` maps 100% of scanned keys, and confirm forward logits, combined loss, and all decoder layer gradients match `scan_layers=False`.
2. **S.2 Asymmetric Position Grid & Negative Control** *(if the model uses multi-axis positions)*:
   - Test a non-square grid where every axis has a distinct coordinate range, plus a negative control that swaps two axes in `decoder_positions` and **must fail**, proving sensitivity to axis ordering.
3. **S.3 Multi-Chip 2D Mesh Sharding Parity** *(requires ≥ 4 chips; skip and note it on single-chip VMs)*:
   - With `ici_fsdp_parallelism=2, ici_tensor_parallelism=2`, inspect physical `PartitionSpec`s and verify 2D-sharded forward logits, loss, and backward gradients match the unsharded mini model.
4. **S.4 Production Mixed-Precision `bfloat16` Sanity Check** *(always)*:
   - Run forward + backward with `dtype="bfloat16"`, `weight_dtype="float32"`, `matmul_precision="default"` and verify zero `NaN`/`Inf` values, `100%` top-1 logit agreement, and gradient `cosine_sim` above the §1.3 threshold across all gradient groups.

> [!TIP]
> **Test Layout**: Implement the core dimensions in `tests/integration/{model_name}_mini_verification_test.py` and the supplemental checks in `tests/integration/{model_name}_mini_supplemental_test.py`. Keep each script self-contained and record JSON evidence (metrics per check) to `/tmp/{model_name}_mini_verification_evidence.json` so the handover checklist can be filled from it.

---

## 4. Standard E2E Golden Logits & 16-Token Autoregressive Decode Check

In addition to the 6-dimension mini-verification suite above, run the standard MaxText CLI logits checker and 16-token greedy autoregressive decode check.

### 4.1 Generate Golden Mini Logits & Golden Decode
For **text-only** models, generate reference golden logits and a golden 16-token greedy decode from the local mini HF model in `/dev/shm/hf_mini/{model_name}_mini`:

```python
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

user = os.environ.get("USER", "user")
mini_model_path = "/dev/shm/hf_mini/{model_name}_mini"
tokenizer = AutoTokenizer.from_pretrained(mini_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mini_model_path, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
)
model.eval()

prompt = "I love to"
inputs = tokenizer(prompt, return_tensors="pt")

# 1. Golden Forward Pass Logits
with torch.no_grad():
  outputs = model(**inputs)
  golden_logits = outputs.logits[0].detach().cpu().numpy().tolist()

# Save to golden_mini_logits.jsonl for forward_pass_logit_checker
os.makedirs(f"/dev/shm/{user}", exist_ok=True)
with open(f"/dev/shm/{user}/golden_mini_logits.jsonl", "w") as f:
  f.write(json.dumps({"prompt": prompt, "tokens": inputs.input_ids[0].tolist(), "logits": golden_logits}) + "\n")

# 2. Golden 16-Token Autoregressive Decode (Greedy)
with torch.no_grad():
  gen_tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False)
  golden_decoded_text = tokenizer.decode(gen_tokens[0])
  print("HF Golden 16-Token Output:", repr(golden_decoded_text))
  print("HF Golden Token IDs:", gen_tokens[0].tolist())
```

For **multimodal** models (`use_multimodal=true`), the logits checker requires pre-generated golden logits from `tests/assets/logits_generation/generate_hf_golden_logits.py`:

```bash
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
  --model-id={hf_model_id} --hf-model-path=/dev/shm/hf_mini/{model_name}_mini \
  --prompts="Describe this image." --image-paths=/path/to/test.jpg --apply-chat-template \
  --hf-load-dtype=float32 --output-path=/dev/shm/$USER/golden_mini_logits.jsonl
```

### 4.2 Run MaxText Forward Pass Logits Checker on TPU VM
Execute the logits checker directly from the MaxText root directory on your TPU VM. MaxText config is passed as `base.yml` + `key=value` overrides; checker options are `--flags`, and **`--max_kl_div` (or `--atol`) is required**:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
  model_name={model_name} \
  load_parameters_path=$MINI_CKPT \
  $MINI_OVERRIDES \
  scan_layers=$MINI_SCAN \
  per_device_batch_size=1 \
  max_prefill_predict_length=16 \
  max_target_length=32 \
  dataset_type=synthetic \
  --hf_model_path=/dev/shm/hf_mini/{model_name}_mini \
  --golden_logits_path=/dev/shm/$USER/golden_mini_logits.jsonl \
  --max_kl_div=1e-3
```

#### Key Evaluation Metrics:
1. **Average KL Divergence per token** ($D_{KL}(P_{golden} \parallel Q_{model})$): Expected range $< 10^{-3}$.
2. **Maximum KL Divergence**: Target $\le 10^{-2}$.
3. **Top-K Token Overlap & Jaccard Similarity**: Confirms top token distributions align.

### 4.3 Run MaxText Autoregressive Decode Check (16 Tokens)
Run greedy decoding on MaxText using the mini converted checkpoint to generate 16 additional tokens:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
  model_name={model_name} \
  tokenizer_type=huggingface \
  tokenizer_path=/dev/shm/hf_mini/{model_name}_mini \
  load_parameters_path=$MINI_CKPT \
  $MINI_OVERRIDES \
  scan_layers=$MINI_SCAN \
  per_device_batch_size=1 \
  prompt="I love to" \
  max_prefill_predict_length=8 \
  max_target_length=24 \
  decode_sampling_strategy=greedy
```

#### Success Criteria for Decode:
- **Gibberish Match**: Because the mini model is a truncated or random-initialized subset of the full architecture, the output text will naturally be gibberish.
- **The verification passes if MaxText and HuggingFace generate the exact same token IDs and identical gibberish string**.
- This strictly verifies that KV cache management, positional index updates, causal attention masking, and autoregressive step execution in the JAX decoder are bug-free.

---

## 5. Iterative Debugging Loop (Closing the Gap)
If any forward boundary, loss component, backward gradient group, or decode check diverges:

> [!IMPORTANT]
> **Free Shared Memory Space**:
> If checks fail due to checkpoint conversion bugs, immediately delete the converted Orbax checkpoint from `/dev/shm/$USER/checkpoints` using `rm -rf <path_to_checkpoint>` before fixing the bug and running conversion again.

### Step-by-Step Debugging Guide:
1. **Is it off from token 0 / layer 0?**
   - Check token embedder weights, transposition, embedding scaling, and input ID mapping.
2. **Is it off at the Multimodal Scatter Boundary?** *(if multimodal)*
   - Check that the model's multimodal placeholder token IDs are registered in its `src/maxtext/multimodal/processor_*.py`, and that `model_name` is included wherever MaxText gates multimodal behavior by model name (grep for an existing multimodal model name to find these allowlists).
3. **Is it off in an attention, MLP, or auxiliary projection in `float32`?**
   - Check RMSNorm epsilon and zero-centered offset (`1.0 + weight` vs `weight`) against HF `config.json` and `modeling_*.py`.
   - Verify Q/K/V projection transpositions and head dimension layout.
   - Check that every `DenseGeneral` layer (including auxiliary/MTP projections) passes `matmul_precision=cfg.matmul_precision` so the TPU MXU does not silently truncate `float32` verification to `bfloat16`.
   - Rotary Embeddings (RoPE): Check `rope_theta`, partial rotary dimensions, any multi-axis section split, and scaling settings, and verify whether `permute_to_match_maxtext_rope` is needed.
4. **Is it off only on Left-Padded or Packed Sequences?**
   - Use an attribution probe to check whether recurrent/conv layers require a pre-`conv1d` padding mask or whether attention masks properly apply segment IDs.
5. **Is it off only during decode (prefill logits match)?**
   - KV Cache: Check single-step decode attention mask, cache update indexing, and position increments; for recurrent layers, check conv/recurrent state carry-over.
6. **MoE routing** *(if MoE)*:
   - Check expert gate weights, expert weight permutation, and top-k gating probability normalization (`norm_topk_prob`).

### Deep Verification via Intermediate Activations Mapping:
If discrepancies persist:
1. Run both HF PyTorch and MaxText with $N=1$ layer (one layer of the failing type) and feed identical input tokens.
2. Capture activations at each sub-layer (HF forward hooks vs. MaxText `sow` / intermediate outputs):
   - Input embeddings
   - Pre-attention norm output
   - Attention Q, K, V projections and attention output
   - Post-attention / MLP norm output
   - MLP intermediate and output projections
   - Final norm and logits
3. Compare intermediate tensors side-by-side to pinpoint the exact failing operation.

---

## 6. Final Session Wrap-up & Handover Document
Once the mini subset passes all applicable core dimensions, supplemental checks, E2E logits, and decode validation:
1. Finalize `{model_name}_handover.md` in the MaxText root directory.
2. Record:
   - Final status of all 4 phases (`DONE`).
   - Quantitative evidence from the 6 Core Mini-Verification Dimensions and Supplemental Checks (mark non-applicable items `N/A` with a reason).
   - Summary of all bugs found during bring-up and their fixes (distinguishing model-specific bring-up bugs from shared MaxText behaviors).
   - Full executable commands for building the mini model, converting, running mini-verification tests, checking logits, and decoding.

---

## 7. Phase 4 Mini-Verification Checklist Template (For `{model_name}_handover.md`)

Copy and populate the following checklist inside `{model_name}_handover.md` during Phase 4. Mark items that do not apply to your architecture as `N/A` with a one-line reason.

```markdown
### Phase 4 Mini-Model Verification Checklist (`{model_name}`)

#### Gate 0: Mini Model & Weight Parity
- [ ] **Representative Mini Model**: `<decoder layers kept (cycle description)>` + multimodal encoder (`<Yes/No/N/A>`) + auxiliary/MTP heads (`<Yes/No/N/A>`)
- [ ] **Mini Model Construction**: `<approach (e.g. A: sliced real HF weights + to_maxtext)>` and why; full → mini sizes (`<layers / experts / widths>`); `$MINI_OVERRIDES = <...>`; `<num_tensors>` parameter tensors
- [ ] **Weight Parity**: HF vs. MaxText mini checkpoint `max_abs_diff = <val>` (target `0.0`, same dtype)

#### Core Dimension 1: Design Challenging Inputs Deliberately
- [ ] **1.1 Sequence & Block/Chunk Boundaries**: Tested $S \in \{<lengths>\}$ straddling `<boundaries: conv kernel K, chunk C, sliding window W, ...>` (`min_cosine_sim = <val>`, `top1 = <val>%`)
- [ ] **1.2 Heterogeneous Padded Batches**: Tested unpadded, right-padded, and left-padded sequences (`cosine_sim = <val>`; attribution probe recorded if shared layer masks differ)
- [ ] **1.3 Packed-Document Boundaries (`segment_ids`)**: Tested multi-segment `[1..1, 2..2]` with per-segment resetting positions; verified isolated block parity (`cosine_sim = <val>`) and zero cross-document leakage
- [ ] **1.4 Chat Templates & Special/Boundary Tokens**: Tested model chat framing, special tokens, and boundary token IDs `0` and `vocab_size - 1` (`cosine_sim = <val>`)
- [ ] **1.5 Multimodal + Position-ID Inputs** *(if VLM/Omni)*: Tested visual patch grid with explicit position IDs (`cosine_sim = <val>`, `top1 = <val>%`)

#### Core Dimension 2: Check Forward Outputs and Masks at Meaningful Boundaries
- [ ] **2.1 Embedding & Scatter Boundaries**:
  - [ ] Token embeddings (`cosine_sim = <val>`)
  - [ ] *(if VLM/Omni)* Patch + Pos Embed (`cosine_sim = <val>`)
  - [ ] *(if VLM/Omni)* Vision Block Stack + Merger/Projector (`cosine_sim = <val>`)
  - [ ] *(if VLM/Omni)* Post-Scatter Multimodal Input Embeddings (`cosine_sim = <val>`)
- [ ] **2.2 Recurrent / Linear-Attention Sub-Stages** *(if hybrid)*: `<list sub-stages>` (`min_cosine_sim = <val>`)
- [ ] **2.3 Full-Attention Sub-Stages**: `<list sub-stages: projections, qk-norm, RoPE, mask, gate, o_proj>` (`min_cosine_sim = <val>`)
- [ ] **2.4 MLP / MoE Block Boundary**: Gate/up projections (`wi_0`, `wi_1`), activation, and down projection (`wo`) (`cosine_sim = <val>`)
- [ ] **2.5 Final Norm & Main LM Head Logits**: `decoder_norm` and `logits_dense` (`cosine_sim = <val>`, `top1 = <val>%`)
- [ ] **2.6 Auxiliary / MTP Head Sub-Stages** *(if MTP)*: `<list sub-stages>` (`min_cosine_sim = <val>`)

#### Core Dimension 3: MoE / Dense Coverage Requirements
- [ ] **3.1 Architecture Coverage (`<Dense or MoE>`)**:
  - *If MoE*: Uniform, collapsed, capacity-boundary, and forced routing verified (`cosine_sim = <val>`, `moe_lb_loss diff = <val>`)
  - *If Dense*: Dense MLP intermediate & output activations verified (`cosine_sim = <val>`); confirmed `num_experts == 1` and `moe_lb_loss == 0.0`

#### Core Dimension 4: Verify Every Enabled Head and the Actual Training Objective
- [ ] **4.1 Pre-Aggregation Loss Components Exposed**: Inspected `aux` keys `xent_sum`, `total_weights`, `z_loss`, `mtp_loss`, `moe_lb_loss` and total loss
- [ ] **4.2 Objective Ablation Matrix**:
  - [ ] `Config_A` (Main CE Only): `abs_diff = <val>`
  - [ ] `Config_B` (Main CE + Z-Loss): `abs_diff = <val>`, `z_loss_diff = <val>`
  - [ ] `Config_C` (MTP Loss Isolated) *(if MTP)*: `mtp_loss_diff = <val>`
  - [ ] `Config_D` (Combined Production Objective): `abs_diff = <val>`, `rel_diff = <val>`

#### Core Dimension 5: Make Backward Checks Sensitive and Independent
- [ ] **5.1 End-to-End Parameter Group Gradients**: Verified all `<K>` parameter groups from input embeddings through decoder blocks, `lm_head`, and auxiliary heads (`min_cosine_sim = <val>`)
- [ ] **5.2 Isolated Block-Level Upstream-Gradient (VJP) Probes**: Injected deterministic synthetic cotangent $G$ into each distinct block type; verified $\partial L / \partial x_{\text{in}}$ and $\partial L / \partial W_{\text{block}}$ (`min_cosine_sim = <val>`)

#### Core Dimension 6: Test Actual Multi-Step Updates and State Propagation
- [ ] **6.1 Trainable / Frozen Parameter Selection**: Verified frozen groups (`<e.g. freeze_vision_encoder_params=True>`) have `max|grad(frozen)| == 0.0`, `max|grad(trainable)| > 0.0`, and bitwise weight invariance across steps
- [ ] **6.2 Multi-Step Stateful AdamW Loss Trajectory**: Ran 3 steps with changing batches (`Step 1 diff = <val>`, `Step 2 diff = <val>`, `Step 3 diff = <val>`)
- [ ] **6.3 End-of-Step-3 Weights & Optimizer Moments**: Verified updated weights $W^{(3)}$ (`min_cosine_sim = <val>`) and AdamW moments $m^{(3)}, v^{(3)}$ (`min_cosine_sim = <val>`) across all subsystems

#### Supplemental Production-Readiness Checks
- [ ] **S.1 `scan_layers=True` Parity**: `PARAM_MAPPING(..., scan_layers=True)` coverage (`<mapped>/<total>`), forward logits (`cosine_sim = <val>`), loss (`diff = <val>`), and gradients (`max_abs_diff = <val>`)
- [ ] **S.2 Asymmetric Position Grid & Negative Control** *(if multi-axis positions)*: Non-square grid (`cosine_sim = <val>`, `top1 = <val>%`) + swapped-axis negative control fails as expected
- [ ] **S.3 Multi-Chip 2D Mesh Sharding Parity** *(≥ 4 chips)*: Verified physical `PartitionSpec`s + sharded vs. unsharded forward logits (`cosine_sim = <val>`), loss (`diff = <val>`), and gradients (`min_cosine_sim = <val>`)
- [ ] **S.4 Production Mixed-Precision `bfloat16` Sanity Check**: `0` NaNs/Infs, `100%` top-1 logit agreement, `loss rel_diff = <val>`, `min_grad_cosine_sim = <val>`

#### Standard CLI Logits & 16-Token Autoregressive Decode Check
- [ ] **CLI `forward_pass_logit_checker`**: `avg_kl_div = <val>` ($\le 10^{-3}$), `max_kl_div = <val>` ($\le 10^{-2}$)
- [ ] **CLI 16-Token Autoregressive Decode (`maxtext.inference.decode`)**: Exact 16-token ID and string match against HuggingFace greedy decode
```
