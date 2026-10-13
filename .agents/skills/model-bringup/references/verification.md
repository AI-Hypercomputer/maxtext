# Comprehensive Mini-Model Verification, E2E Logits & Decode Verification

This guide covers Phase 4 of the MaxText model bring-up workflow: verifying a representative trimmed (mini) version of the target model across **forward boundaries, enabled heads/objectives, backward gradients, and multi-step stateful optimization**, alongside **E2E forward-pass logits validation** and **16-token autoregressive decode checks** before full-scale model verification.

---

## 1. Fast Mini-Subset Verification Workflow

Always verify a **representative trimmed version** ($N$ layers + any multimodal vision/audio encoders + any auxiliary heads such as Multi-Token Prediction) first using real weights in `/dev/shm/hf_mini/{model_name}_{N}layers` and `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax`.

### 1.1 Slicing & Round-Trip Checkpoint Parity (`slice_model.py` + `to_huggingface.py`)
1. **Pick a Representative Trimmed Model**:
   - Choose $N$ layers that cover one complete architectural cycle (e.g., $N=4$ for a 3:1 hybrid linear-attention + full-attention cycle, or layers covering both dense and MoE blocks), while preserving the Vision Encoder / Projector and Multi-Token Prediction (MTP) layers if present.
2. **Create Matched MaxText & HuggingFace Mini Checkpoints**:
   - Use the MaxText slicing tool (`src/maxtext/checkpoint_conversion/slicing/slice_model.py`) to slice the MaxText Orbax checkpoint to $N$ decoder layers, and convert it back to HuggingFace format via `src/maxtext/checkpoint_conversion/to_huggingface.py` (or convert the sliced HF checkpoint via `to_maxtext.py`).
   - **Gate 0 Requirement**: Verify that every parameter tensor across text decoder, vision encoder/merger, and auxiliary heads matches between the sliced PyTorch/HF and MaxText mini checkpoints with **`max_abs_diff == 0.0`**.

### 1.2 Strategy First: Architecture-Specific Dimension Analysis & Attribution Principle
Before executing tests, define a verification strategy tailored to the target model's architecture:
- **Identify the Architectural Dimensions That Matter**:
  - *Hybrid / Recurrent Attention* (e.g., Gated DeltaNet + Full Attention with Output Gate)
  - *Dense vs. Mixture-of-Experts (MoE)* (e.g., dense SwiGLU MLP vs. routed + shared experts)
  - *Multimodal Encoders & Positional Embeddings* (e.g., 3D Conv ViT + Spatial Merger + 3D Multimodal RoPE `[T, H, W]`)
  - *Auxiliary Heads & Objectives* (e.g., Multi-Token Prediction `mtp_block`, Z-loss, MoE load-balancing loss)
- **Attribution Principle (Bringup Bug vs. Underlying Shared Behavior)**:
  - Whenever a discrepancy appears on challenging inputs (such as left-padding or packed segments in recurrent/conv layers, or multi-frame `video_mask` in ViT), write a **targeted attribution probe** (e.g., isolating a single layer or injecting a 1-line mask hook) to determine whether the discrepancy was introduced by the new model bring-up or is an underlying behavior of a shared MaxText module.

---

## 2. The 6 Core Mini-Verification Dimensions

The mini PyTorch/HuggingFace model and the mini MaxText/JAX model must behave identically (up to floating-point numerical precision in `float32` with `matmul_precision="highest"`) across all **6 verification dimensions**:

### Dimension 1: Design Challenging Inputs Deliberately
Test synthetic inputs specifically crafted to stress boundary conditions rather than only simple single-sentence prompts:
- **Test sequence and block/chunk boundaries**:
  - Test sequence lengths straddling causal `conv1d` kernel sizes ($K$) and recurrent/linear-attention chunk sizes ($C$), e.g., $S \in \{K-1, K, K+1, C-1, C, C+1\}$ (such as $S \in \{3, 4, 5, 63, 64, 65\}$ for $K=4, C=64$).
  - Verify that partial chunks and single-token remainder chunks (`65 = 64 + 1`) match with `cosine_sim >= 0.999999` and `100%` top-1 agreement.
- **Test heterogeneous batches**:
  - Construct batches with variable valid lengths under unpadded, right-padded, and left-padded layouts (`decoder_segment_ids != 0`).
  - If recurrent/conv layers do not consume `decoder_segment_ids`, use a pre-`conv1d` mask attribution probe to verify exact parity (`cosine_sim = 1.0`).
- **Test packed-document boundaries**:
  - Construct multi-document packed sequences (`decoder_segment_ids = [1..1, 2..2]`) with per-segment resetting `decoder_positions = [0..L_1-1, 0..L_2-1]`.
  - Verify both isolated attention-block parity under block-diagonal causal masks and zero cross-document leakage when Document 2 tokens are perturbed.
- **Test different chat templates and special tokens**:
  - Include model-specific chat framing (`<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`, `<think>`, `</think>`) and vocabulary boundary token IDs (`0` and `vocab_size - 1`).
- **Test multimodal & 3D MRoPE inputs** (for VLMs / Omni models):
  - Feed raw synthetic visual patches (`pixel_values` / `encoder_images`) with explicit 3D `[T, H, W]` position IDs (`mrope_section`).

### Dimension 2: Check Forward Outputs and Masks at Meaningful Boundaries
Compare **both component-level intermediate outputs and complete small-model logits** (`cosine_sim`, `rel_l2`, `max_abs_diff`, and `top1_agreement`):
1. **Vision Encoder & Multimodal Scatter Boundaries** (if multimodal):
   - *Boundary 1a*: Vision Patch + Learned Position Embeddings (`patch_embed` + `pos_embed_interpolate`).
   - *Boundary 1b*: Vision ViT Block Stack + Spatial Merger output (`pooler_output`).
   - *Boundary 1c*: Post-scatter Multimodal Input Embeddings (`place_multimodal_embeddings` into `<|image_pad|>` token slots — catches silent special-token ID mismatches before the decoder).
2. **Linear / Recurrent Attention Sub-Stages** (if hybrid/recurrent):
   - Input RMSNorm $\rightarrow$ fused/split projections (`in_proj_qkvz`, `in_proj_ba`) $\rightarrow$ depthwise causal `conv1d` + SiLU $\rightarrow$ recurrent delta rule ($g, \beta$ gates & state update) $\rightarrow$ gated RMSNorm $\rightarrow$ `out_proj`.
3. **Full-Attention Sub-Stages**:
   - `q_norm` and `k_norm` per-head RMSNorm $\rightarrow$ RoPE / 3D MRoPE rotation $\rightarrow$ causal/segment mask application $\rightarrow$ gated attention output (`sigmoid(gate) * attn_out`) $\rightarrow$ `o_proj`.
4. **MLP / MoE Sub-Stages**:
   - Post-attention RMSNorm $\rightarrow$ gate/up projections (`wi_0`, `wi_1`) & activation $\rightarrow$ down projection (`wo`).
5. **Final Norm, Main LM Head, and Auxiliary (MTP) Sub-Stages**:
   - Final decoder RMSNorm $\rightarrow$ `logits_dense` (`lm_head`) $\rightarrow$ MTP `roll_and_mask` $\rightarrow$ MTP `pre_fc_norm_embedding` & `pre_fc_norm_hidden` $\rightarrow$ MTP `fc` projection $\rightarrow$ MTP transformer block $\rightarrow$ MTP `final_norm` $\rightarrow$ shared `lm_head` MTP logits.

### Dimension 3: Give MoE / Dense Its Own Coverage Requirements
- **If the model is MoE**:
  - **Cover routing patterns deliberately**: Test uniform routing, skewed/collapsed routing (all tokens to 1 expert), capacity-boundary routing, and padded-token routing.
  - **Use forced routing if needed**: Override router logits deterministically to isolate expert weight permutation (`wi_0`, `wi_1`, `wo`), `norm_topk_prob`, shared-expert sigmoid gating, and `moe_lb_loss`.
- **If the model is Dense**:
  - Verify dense SwiGLU/GeGLU intermediate activations (`act(wi_0(x)) * wi_1(x)`) and output projection (`wo`) at the sub-module boundary.
  - Confirm `num_experts == 0` and `total_moe_lb_loss == 0.0` in `aux` metrics (zero spurious routing overhead).

### Dimension 4: Verify Every Enabled Head and the Actual Training Objective
- **Inspect all enabled outputs**: Main LM head logits (`[B, S, V]`), MTP head logits (`[B, S, V]`), and multimodal projector outputs (`[N_merged, D]`).
- **Expose loss components before aggregation**: Inspect `total_loss`, `main_ce_loss`, `total_z_loss`, `mtp_loss`, `total_moe_lb_loss`, and `total_weights`.
- **Test objectives separately and together** across a 4-way ablation matrix:
  1. `Config_A` (Main Cross-Entropy Only): `mtp_loss_scaling_factor=0.0, z_loss_multiplier=0.0`
  2. `Config_B` (Main Cross-Entropy + Z-Loss): `mtp_loss_scaling_factor=0.0, z_loss_multiplier=1e-4`
  3. `Config_C` (MTP Objective Isolated): `mtp_loss_scaling_factor=1.0, z_loss_multiplier=0.0` (compare `aux["mtp_loss"]` directly against PyTorch MTP cross-entropy)
  4. `Config_D` (Combined Production Objective): `mtp_loss_scaling_factor=0.1, z_loss_multiplier=1e-4` with multimodal inputs.

### Dimension 5: Make Backward Checks Sensitive and Independent
- **Compare gradients per parameter group and input**:
  - Run `jax.value_and_grad` on the combined multimodal + auxiliary objective (`Config_D`) and compare against PyTorch `.backward()` across every parameter group from input to output (e.g., `vision.patch_embed`, `vision.blocks[-1]`, `vision.merger`, `token_embedder`, recurrent `A_log`/`dt_bias`/`conv1d`, full-attention `qkv`/`q_norm`/`k_norm`, `mlp.wi_0`/`wi_1`/`wo`, `decoder_norm`, `logits_dense`, and `mtp_block`).
- **Add block-level upstream-gradient (VJP) probes**:
  - Generate a deterministic synthetic upstream cotangent $G \sim \mathcal{N}(0, 1)$ (`seed=12345`) and compute the isolated Vector-Jacobian Product $\langle f_{\text{block}}(x, W), G \rangle$ for each distinct block type (e.g., Linear-Attention Layer 0, Full-Attention Layer 3, MTP Layer 1).
  - Verify that both the input activation gradient ($\partial L / \partial x_{\text{in}}$) and local parameter gradients ($\partial L / \partial W_{\text{block}}$) match PyTorch with `cosine_sim >= 0.999999` completely independently of downstream layers or loss scaling.

### Dimension 6: Test Actual Multi-Step Updates and State Propagation
- **Run several real optimizer updates with changing batches**:
  - Initialize an identical unclipped stateful AdamW optimizer (`optax.inject_hyperparams(optax.adamw)(learning_rate=1e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01)` vs. `torch.optim.AdamW`) and run **3 consecutive training steps** on **3 distinct changing batches**.
- **Compare state at the end of the complete step**:
  - Compare the per-step loss trajectory ($L^{(1)}, L^{(2)}, L^{(3)}$), the updated weights ($W^{(3)}$), and the first and second optimizer moments ($m^{(3)}$ / `mu` and $v^{(3)}$ / `nu`) across all subsystems (`cosine_sim > 0.9999`).
- **Verify trainable/frozen parameter selection**:
  - Test config flags such as `freeze_vision_encoder_params=True` (`jax.lax.stop_gradient`) and masked optimizer parameter selection: assert `max |grad(frozen)| == 0.0`, `max |grad(trainable)| > 0.0`, and exact bitwise weight invariance ($W_{\text{frozen}}^{(3)} \equiv W_{\text{frozen}}^{(0)}$, `max_abs_diff == 0.0`) after 3 optimizer steps.

---

## 3. Supplemental Production-Readiness Checks (Bridging Mini Model to Full Scale)

Before concluding mini-model verification, run 4 fast supplemental checks to ensure production flags do not introduce regressions:
1. **`scan_layers=True` (`ScannableBlock`) vs. `scan_layers=False` Parity**:
   - Verify `PARAM_MAPPING(..., scan_layers=True)` maps 100% of scanned keys, and confirm forward logits, combined loss, and all decoder layer gradients match `scan_layers=False` (`cosine_sim = 1.0`).
2. **Asymmetric 3D MRoPE Grid ($T \neq H \neq W$) & Axis-Swap Negative Control**:
   - Test a non-square grid (e.g., `grid_thw = [1, 4, 6]`) where all 3 axes have distinct coordinate ranges, plus a negative control swapping $T \leftrightarrow W$ in `decoder_positions` to prove `mrope_section` sensitivity to axis ordering.
3. **Multi-Chip 2D Mesh Sharding Parity (`ici_fsdp_parallelism=2, ici_tensor_parallelism=2`)**:
   - Inspect physical `PartitionSpec`s across all TPU chips and verify 2D-sharded forward logits, loss, and backward gradients match the unsharded mini model (`cosine_sim = 1.0`).
4. **Production Mixed-Precision `bfloat16` Sanity Check**:
   - Run forward + backward with `dtype="bfloat16"`, `dtype_mm="bfloat16"`, `weight_dtype="float32"`, `matmul_precision="default"` and verify zero `NaN`/`Inf` values, `100%` top-1 logit agreement, and `cosine_sim > 0.998` across all gradient groups.

> [!TIP]
> **Reference Implementation**: See `tests/integration/qwen3_8_27b_mini_verification_test.py` (6 Core Dimensions) and `tests/integration/qwen3_8_27b_mini_supplemental_test.py` (4 Supplemental Production-Readiness Checks) for complete, self-contained reference scripts that record JSON evidence to `/tmp/{model_name}_mini_verification_evidence.json`.

---

## 4. Standard E2E Golden Logits & 16-Token Autoregressive Decode Check

In addition to the 6-dimension mini-verification suite above, run the standard MaxText CLI logits checker and 16-token greedy autoregressive decode check.

### 4.1 Generate Golden Mini Logits & Golden Decode
Generate standard reference golden logits and golden 16-token autoregressive decode using the local mini PyTorch HuggingFace model in `/dev/shm/hf_mini/{model_name}_xlayers`:

```python
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

user = os.environ.get("USER", "user")
mini_model_path = "/dev/shm/hf_mini/{model_name}_xlayers"
tokenizer = AutoTokenizer.from_pretrained(mini_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mini_model_path, torch_dtype=torch.float32, trust_remote_code=True)
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

### 4.2 Run MaxText Forward Pass Logits Checker on TPU VM
Execute the logits checker script directly from the MaxText root directory on your TPU VM:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m tests.utils.forward_pass_logit_checker \
  --model_name={model_name} \
  --maxtext_model_path=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax \
  --golden_logits_path=/dev/shm/$USER/golden_mini_logits.jsonl \
  --base_num_decoder_layers={N}
```

#### Key Evaluation Metrics:
1. **Average KL Divergence per token** ($D_{KL}(P_{golden} \parallel Q_{model})$): Expected range $< 10^{-3}$.
2. **Maximum KL Divergence**: Target $\le 10^{-2}$.
3. **Top-K Token Overlap & Jaccard Similarity**: Confirms top token distributions align.

### 4.3 Run MaxText Autoregressive Decode Check (16 Tokens)
Run greedy decoding on MaxText using the mini converted checkpoint to generate 16 additional tokens:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.inference.decode \
  src/maxtext/configs/base.yml \
  model_name={model_name} \
  load_parameters_path=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax \
  base_num_decoder_layers={N} \
  prompt="I love to" \
  max_target_length=20 \
  temperature=0.0
```

#### Success Criteria for Decode:
- **Gibberish Match**: Because the model only has $N$ layers (an incomplete model), the output text will naturally be gibberish.
- **The verification passes if MaxText and HuggingFace generate the exact same token IDs and identical gibberish string**.
- This strictly verifies that KV cache management, positional index updates, causal attention masking, and autoregressive step execution in the JAX decoder are bug-free.

---

## 5. Iterative Debugging Loop (Closing the Gap)
If any forward boundary, loss component, backward gradient group, or decode check diverges:

> [!IMPORTANT]
> **Free Shared Memory Space**:
> If checks fail due to checkpoint conversion bugs, immediately delete the converted Orbax checkpoint from `/dev/shm/$USER/checkpoints` using `rm -rf <path_to_checkpoint>` before fixing the bug and running conversion again.

### Step-by-Step Debugging Guide:
1. **Is it off at the Multimodal Scatter Boundary (Boundary 1c)?**
   - Check that the model's `<|image_pad|>` / `<|video_pad|>` token IDs are registered in `src/maxtext/multimodal/processor_*.py` (e.g. `QWEN_SPECIAL_TOKEN_CONFIGS`), and that `model_name` is in the multimodal allowlists in `nnx_decoders.py`, `processor.py`, and `types.py`.
2. **Is it off in a Linear / Full Attention or MTP Projection in `float32`?**
   - Check if RMSNorm epsilon and zero-centered offset (`1.0 + weight` vs `weight`) match HF `config.json` and `modeling_*.py`.
   - Check that every `DenseGeneral` layer (including auxiliary/MTP projections) passes `matmul_precision=cfg.matmul_precision` so TPU MXU does not silently truncate `float32` verification to `bfloat16`.
   - Rotary Embeddings (RoPE / 3D MRoPE): Check `rope_theta`, `partial_rotary_factor`, and `mrope_section`, and verify whether `permute_to_match_maxtext_rope` is needed.
3. **Is it off only on Left-Padded or Packed Sequences?**
   - Use an attribution probe to check whether recurrent/conv layers (such as `GatedDeltaNet`) require a pre-`conv1d` padding mask or whether attention masks properly apply segment IDs.

---

## 6. Final Session Wrap-up & Handover Document
Once the mini subset passes all 6 core verification dimensions, the 4 supplemental checks, E2E logits, and decode validation:
1. Finalize `{model_name}_handover.md` in the MaxText root directory.
2. Record:
   - Final status of all 4 phases (`DONE`).
   - Quantitative evidence from the 6 Core Mini-Verification Dimensions and 4 Supplemental Checks.
   - Summary of all bugs found during bring-up and their fixes (distinguishing model-specific bring-up bugs from shared MaxText behaviors).
   - Full executable CLI commands for slicing, converting, running mini-verification tests, checking logits, and decoding.

---

## 7. Phase 4 Mini-Verification Checklist Template (For `{model_name}_handover.md`)

Copy and populate the following checklist inside `{model_name}_handover.md` during Phase 4:

```markdown
### Phase 4 Mini-Model Verification Checklist (`{model_name}`)

#### Gate 0: Slicing & Round-Trip Checkpoint Parity
- [ ] **Representative Trimmed Subset Selected**: `$N$` decoder layers (`<cycle description>`) + Vision Encoder (`<Yes/No>`) + Auxiliary/MTP Heads (`<Yes/No>`)
- [ ] **MaxText Orbax Slicing (`slice_model.py`)**: Sliced from `$M$` layers to `$N$` layers (`<num_tensors>` total parameter tensors)
- [ ] **Reverse Conversion (`to_huggingface.py`)**: Exported sliced Orbax checkpoint to HuggingFace safetensors with `max_abs_diff = <val>` (target `0.0`)

#### Core Dimension 1: Design Challenging Inputs Deliberately
- [ ] **1.1 Sequence & Block/Chunk Boundaries**: Tested $S \in \{<lengths>\}$ straddling conv kernel $K$ and recurrent chunk size $C$ (`min_cosine_sim = <val>`, `top1 = <val>%`)
- [ ] **1.2 Heterogeneous Padded Batches**: Tested unpadded, right-padded, and left-padded sequences (`cosine_sim = <val>`; attribution probe recorded if shared layer masks differ)
- [ ] **1.3 Packed-Document Boundaries (`segment_ids`)**: Tested multi-segment `[1..1, 2..2]` with per-segment resetting positions; verified isolated block parity (`cosine_sim = <val>`) and zero cross-document leakage
- [ ] **1.4 Chat Templates & Special/Boundary Tokens**: Tested model chat framing, multimodal delimiters, and boundary token IDs `0` and `vocab_size - 1` (`cosine_sim = <val>`)
- [ ] **1.5 Multimodal + 3D MRoPE Inputs** *(if VLM/Omni)*: Tested visual patch grid with explicit 3D `[T, H, W]` coordinates (`cosine_sim = <val>`, `top1 = <val>%`)

#### Core Dimension 2: Check Forward Outputs and Masks at Meaningful Boundaries
- [ ] **2.1 Vision Encoder & Scatter Boundaries** *(if VLM/Omni)*:
  - [ ] Patch + Pos Embed (`cosine_sim = <val>`)
  - [ ] ViT Block Stack + Spatial Merger (`cosine_sim = <val>`)
  - [ ] Post-Scatter Multimodal Input Embeddings (`<|image_pad|>` slots replaced; `cosine_sim = <val>`)
- [ ] **2.2 Recurrent / Linear-Attention Sub-Stages** *(if hybrid)*: Input norm, QKVZ/BA projections, causal `conv1d`, recurrent delta rule, gated RMSNorm, and `out_proj` (`min_cosine_sim = <val>`)
- [ ] **2.3 Full-Attention Sub-Stages**: `q_norm`, `k_norm`, RoPE/3D MRoPE, segment mask, gated attention output, and `o_proj` (`min_cosine_sim = <val>`)
- [ ] **2.4 MLP / MoE Block Boundary**: Gate/up projections (`wi_0`, `wi_1`), activation, and down projection (`wo`) (`cosine_sim = <val>`)
- [ ] **2.5 Final Norm & Main LM Head Logits**: `decoder_norm` and `logits_dense` (`cosine_sim = <val>`, `top1 = <val>%`)
- [ ] **2.6 Auxiliary / MTP Layer Sub-Stages** *(if MTP enabled)*: `roll_and_mask`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `fc` projection, MTP decoder block, `final_norm`, and shared `lm_head` (`min_cosine_sim = <val>`)

#### Core Dimension 3: MoE / Dense Coverage Requirements
- [ ] **3.1 Architecture Coverage (`<Dense or MoE>`)**:
  - *If MoE*: Uniform, collapsed, capacity-boundary, and forced routing verified (`cosine_sim = <val>`, `moe_lb_loss diff = <val>`)
  - *If Dense*: Dense SwiGLU intermediate & output activations verified (`cosine_sim = <val>`); confirmed `num_experts == 0` and `total_moe_lb_loss == 0.0`

#### Core Dimension 4: Verify Every Enabled Head and the Actual Training Objective
- [ ] **4.1 Pre-Aggregation Loss Components Exposed**: Inspected `total_loss`, `main_ce_loss`, `total_z_loss`, `mtp_loss`, `total_moe_lb_loss`
- [ ] **4.2 4-Way Objective Ablation Matrix**:
  - [ ] `Config_A` (Main CE Only): `abs_diff = <val>`
  - [ ] `Config_B` (Main CE + Z-Loss): `abs_diff = <val>`, `z_loss_diff = <val>`
  - [ ] `Config_C` (MTP Loss Isolated): `mtp_loss_diff = <val>`
  - [ ] `Config_D` (Combined Multimodal + Main CE + MTP + Z-Loss): `abs_diff = <val>`, `rel_diff = <val>`

#### Core Dimension 5: Make Backward Checks Sensitive and Independent
- [ ] **5.1 End-to-End Parameter Group Gradients**: Verified all `<K>` parameter groups from input layer (`vision.patch_embed` / `token_embedder`) through decoder blocks, `lm_head`, and `mtp_block` (`min_cosine_sim = <val>`)
- [ ] **5.2 Isolated Block-Level Upstream-Gradient (VJP) Probes**: Injected deterministic synthetic cotangent $G$ into isolated blocks; verified both input gradient $\partial L / \partial x_{\text{in}}$ and weight gradient $\partial L / \partial W_{\text{block}}$ (`min_cosine_sim = <val>`)

#### Core Dimension 6: Test Actual Multi-Step Updates and State Propagation
- [ ] **6.1 Trainable / Frozen Parameter Selection**: Verified `freeze_vision_encoder_params=True` (`max|grad(frozen)| == 0.0`, `max|grad(trainable)| > 0.0`) and bitwise frozen weight invariance across steps (`max_abs_diff == 0.0`)
- [ ] **6.2 Multi-Step Stateful AdamW Loss Trajectory**: Ran 3 steps with changing batches (`Step 1 diff = <val>`, `Step 2 diff = <val>`, `Step 3 diff = <val>`)
- [ ] **6.3 End-of-Step-3 Weights & Optimizer Moments**: Verified updated weights $W^{(3)}$ (`min_cosine_sim = <val>`) and AdamW moments $m^{(3)}, v^{(3)}$ (`min_cosine_sim = <val>`) across all subsystems

#### Supplemental Production-Readiness Checks
- [ ] **S.1 `scan_layers=True` (`ScannableBlock`) Parity**: `PARAM_MAPPING(..., scan_layers=True)` coverage (`<mapped>/<total>`), forward logits (`cosine_sim = <val>`), loss (`diff = <val>`), and backward gradients (`max_abs_diff = <val>`)
- [ ] **S.2 Asymmetric 3D MRoPE Spatial Grid & Negative Control**: Non-square `[T, H, W]` grid (`cosine_sim = <val>`, `top1 = <val>%`) + swapped $T \leftrightarrow W$ negative control confirming axis-order sensitivity
- [ ] **S.3 Multi-Chip 2D Mesh Sharding Parity (`fsdp=2, tensor=2`)**: Verified physical `PartitionSpec`s + sharded vs. unsharded forward logits (`cosine_sim = <val>`), loss (`diff = <val>`), and gradients (`min_cosine_sim = <val>`)
- [ ] **S.4 Production Mixed-Precision `bfloat16` Sanity Check**: Confirmed `0` NaNs/Infs across all parameters/gradients, `100%` top-1 logit agreement, `loss rel_diff = <val>`, and `min_grad_cosine_sim = <val>`

#### Standard CLI Logits & 16-Token Autoregressive Decode Check
- [ ] **CLI `forward_pass_logit_checker`**: `avg_kl_div = <val>` ($\le 10^{-3}$), `max_kl_div = <val>` ($\le 10^{-2}$)
- [ ] **CLI 16-Token Autoregressive Decode (`maxtext.inference.decode`)**: Exact 16-token ID and string match against HuggingFace greedy decode
```
