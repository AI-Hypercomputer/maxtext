# MaxText Hardware Co-Design & Multi-Pass Precision Emulation

Zero-model-surgery hardware co-design framework and dtype-agnostic **Multi-Pass Precision Emulation** for MaxText across Pre-Training, Supervised Fine-Tuning (SFT), Reinforcement Learning (GRPO/GSPO), Direct Preference Optimization (DPO), Knowledge Distillation, and Evaluation.

> **Quick-Start & OLMo-3 7B Case Study**: See [`multi_pass/multi_pass_emulation.md`](multi_pass/multi_pass_emulation.md) for the quick-start guide, Option-A bitwise-locked baseline setup, and the OLMo-3 7B `FP4 E2M1 (X2)` case study.

## Package Layout

- **`codesign/`** *(General Co-Design Infrastructure & Entry Scripts)*:
  - `emulation_base.py`: Composable `EmulationFeature` plugin base class and registry.
  - `interception.py`: Runtime Linen/NNX operator interception and step/layer context scopes.
  - `hooks.py`: Non-intrusive MaxText/Tunix training, checkpointing, RL rollout, and evaluation hooks.
  - `train.py`, `train_sft.py`, `train_rl.py`, `train_dpo.py`, `train_distill.py`, `eval.py`: Workflow entry points.
  - `tests/test_codesign_hooks.py`: General co-design infrastructure unit tests.
- **`codesign/multi_pass/`** *(Multi-Pass Precision Emulation Subpackage)*:
  - `multipass_kernel.py`: Residual multi-pass tensor decomposition and custom-VJP GEMM kernels.
  - `multipass_config.py`: Multi-pass configuration schema, `DtypeSpec` parser, and hardware presets.
  - `checkpoint_utils.py` & `convert_checkpoint.py`: Checkpoint materialization and per-pass SNR/RMSE inspection.
  - `configs/*.yml`: Ready-to-run YAML presets (`sft_multipass.yml`, `rl_multipass.yml`, `dpo_multipass.yml`, `distill_multipass.yml`, `eval_multipass.yml`, `granular_multipass_rules_example.yml`).
  - `tests/`: Multi-pass unit tests (`test_multipass_emulation.py`) and e2e SFT smoke test (`sft_multipass_emulation_smoke_test.py`).
  - `multi_pass_emulation.md` & `assets/multipass_case_study_curves.svg`: Feature introduction & OLMo-3 7B case study.

---

## 1. Overview & Mathematical Formulation

Multi-pass matrix multiplication decomposes a high-precision source operand $x \in \tau_{\text{src}}$ ($A$) into $K$ sequential lower-precision residual passes $[\tau_0, \tau_1, \dots, \tau_{K-1}]$ ($[B, C, D, \dots]$):

$$r_0 = \text{Quant}_{\tau_{\text{src}}}(x)$$
$$s_k = \text{Scale}(r_k, \tau_k), \qquad q_k = \text{Quant}_{\tau_k}\!\left(\frac{r_k}{s_k}\right), \qquad p_k = s_k \cdot q_k, \qquad r_{k+1} = r_k - p_k \quad (k = 0, \dots, K-1)$$

Given an LHS operand $A \approx \sum_{i=0}^{K_A-1} p^{(A)}_i$ and an RHS operand $B \approx \sum_{j=0}^{K_B-1} p^{(B)}_j$, the matrix product $C = A \cdot B$ is approximated over a configurable set of active cross-term pass pairs $\mathcal{P} \subseteq \{0, \dots, K_A-1\} \times \{0, \dots, K_B-1\}$ accumulated in $\tau_{\text{acc}}$:

$$C_{\text{multipass}} = \text{Cast}_{\tau_{\text{out}}}\!\left( \sum_{(i, j) \in \mathcal{P}} \text{Acc}_{\tau_{\text{acc}}}\!\big( p^{(A)}_i \cdot p^{(B)}_j \big) \right)$$

### Key Design Principles
- **Zero Model Surgery**: All interception is performed via runtime hooks (`hooks.py` and `interception.py`) around `DenseGeneral`, `RoutedMoE`, `AttentionOp`, and `Decoder._apply_layers_sequentially`. No upstream MaxText model definitions or training loops are modified.
- **Fully Dtype-Agnostic**: Decompose any `src_dtype` (`bf16`, `fp32`, `fp16`, `int8`, `auto`, or custom `eXmY`) into any number of passes $K$ with homogeneous or heterogeneous `pass_dtypes` `[B, C, D, ...]` (e.g., `["fp8_e4m3", "fp8_e4m3"]`, `["fp8_e5m2", "fp8_e4m3", "fp4_e2m1"]`, `["bf16", "bf16"]`, `["int4", "uint4"]`, or custom `["e5m3", "e3m2"]`).
- **Configurable Cross-Term Evaluation**:
  - `triangular` (`X3` for $K=2$, `X6` for $K=3$): Executes all $(i, j)$ with $i + j \le \text{max\_pass\_degree}$ (dropping high-order `lo·lo` terms).
  - `full` (`X4` for $K=2$): Executes all $K_A \times K_B$ cross-terms.
  - `asymmetric_lhs` / `asymmetric_rhs` (`X2`): Slices only one operand into $K$ passes and the other into $1$ pass.
  - `karatsuba` (`X3`): Evaluates all $2 \times 2$ cross-terms using 3 matmuls via Karatsuba recombination.
  - `custom`: User-specified list of `(i:j)` pass pairs via `emulate_active_passes`.
- **Independent Forward & Backward Control**: Custom VJP (`bwd_mode="multipass_bwd"`) allows independent multi-pass emulation on forward (`emulate_fwd`), activation gradient (`emulate_dlhs`), and weight gradient (`emulate_drhs`) GEMMs, or Straight-Through Estimator (`bwd_mode="ste"`).
- **Per-Layer & Dynamic Step Scheduling**: Works seamlessly with both `scan_layers=True` and `scan_layers=False`, and supports mid-training activation/deactivation via `emulate_start_step` and `emulate_end_step` without JAX recompilation.

---

## 2. Built-In Algorithm Presets (`emulate_multipass_preset`)

| Preset Name | Source Dtype ($A$) | LHS Passes | RHS Passes | Scaling Mode | Cross-Term Mode | Matmul Passes | Accumulator | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| `F8E4M3_F8E4M3_F32_X1` | `bf16` | `[fp8_e4m3]` | `[fp8_e4m3]` | `per_tensor_pow2` | `full` | 1 | `fp32` | Baseline 1-pass FP8 E4M3 matmul |
| `F8E4M3_F8E4M3_F32_X2` | `bf16` | `[fp8_e4m3]` | `[fp8_e4m3, fp8_e4m3]` | `per_tensor_pow2` | `asymmetric_rhs` | 2 | `fp32` | 2-pass asymmetric FP8 E4M3 (`hh + hl`) |
| `F8E4M3_F8E4M3_F32_X3` | `bf16` | `[fp8_e4m3, fp8_e4m3]` | `[fp8_e4m3, fp8_e4m3]` | `per_tensor_pow2` | `triangular` | 3 | `fp32` | 3-pass symmetric triangular FP8 E4M3 (`hh + hl + lh`, drops `ll`) |
| `F8E4M3_F8E4M3_F32_X4` | `bf16` | `[fp8_e4m3, fp8_e4m3]` | `[fp8_e4m3, fp8_e4m3]` | `per_tensor_pow2` | `full` | 4 | `fp32` | 4-pass full FP8 E4M3 (`hh + hl + lh + ll`) |
| `F8E5M2_F8E5M2_F32_X3` | `bf16` | `[fp8_e5m2, fp8_e5m2]` | `[fp8_e5m2, fp8_e5m2]` | `per_tensor_pow2` | `triangular` | 3 | `fp32` | 3-pass symmetric triangular FP8 E5M2 |
| `BF16_BF16_F32_X2` | `fp32` | `[bf16]` | `[bf16, bf16]` | `none` | `asymmetric_rhs` | 2 | `fp32` | 2-pass asymmetric FP32 $\to$ BF16 |
| `BF16_BF16_F32_X3` | `fp32` | `[bf16, bf16]` | `[bf16, bf16]` | `none` | `triangular` | 3 | `fp32` | 3-pass Henry et al. FP32 $\to$ `[bf16, bf16]` |
| `BF16_BF16_F32_X6` | `fp32` | `[bf16, bf16, bf16]` | `[bf16, bf16, bf16]` | `none` | `triangular` | 6 | `fp32` | 6-pass Henry et al. FP32 $\to$ `[bf16, bf16, bf16]` ($i+j \le 2$) |
| `INT8_INT4_INT32_X2` | `int8` | `[int4, uint4]` | `[int4]` | `bit_shift` | `asymmetric_lhs` | 2 | `int32` | 2-pass asymmetric INT8 $\times$ INT4 via INT4 passes |
| `INT8_INT8_INT32_X3` | `int8` | `[int4, uint4]` | `[int4, uint4]` | `bit_shift` | `karatsuba` | 3 | `int32` | 3-pass Karatsuba INT8 $\times$ INT8 via INT4 passes |
| `INT8_INT8_INT32_X4` | `int8` | `[int4, uint4]` | `[int4, uint4]` | `bit_shift` | `full` | 4 | `int32` | 4-pass exact low-high INT8 $\times$ INT8 via INT4 passes |

> **Note:** Any explicit CLI or YAML flag (e.g. `emulate_scale_mode=per_block_pow2`, `emulate_pass_dtypes=fp8_e5m2,fp4_e2m1`) layers on top of and overrides the corresponding field of `emulate_multipass_preset`.

---

## 3. Complete Flag Reference & Documentation

Every newly introduced CLI / YAML flag is documented below:

| Flag Name | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `emulate_multipass_enabled` | `False` | `bool` | Master switch enabling Multi-Pass Precision Emulation across training or evaluation workflows. |
| `emulate_multipass_preset` | `""` | `str` | Optional named algorithm preset (`F8E4M3_F8E4M3_F32_X1`, `F8E4M3_F8E4M3_F32_X2`, `F8E4M3_F8E4M3_F32_X3`, `F8E4M3_F8E4M3_F32_X4`, `F8E5M2_F8E5M2_F32_X3`, `BF16_BF16_F32_X2`, `BF16_BF16_F32_X3`, `BF16_BF16_F32_X6`, `INT8_INT4_INT32_X2`, `INT8_INT8_INT32_X3`, `INT8_INT8_INT32_X4`). |
| `emulate_src_dtype` | `"bf16"` | `str` | Source tensor dtype $A$ into which each operand is cast/quantized prior to residual pass slicing (`bf16`, `fp32`, `fp16`, `fp8_e4m3`, `int8`, `auto`, or custom `eXmY` / `intB`). |
| `emulate_num_passes` | `2` | `int` | Number of sequential residual passes $K$ per operand (`1`, `2`, `3`, `4`, ...). Automatically inferred as `len(emulate_pass_dtypes)` if a multi-element dtype list is provided without an explicit `emulate_num_passes`. |
| `emulate_pass_dtypes` | `"fp8_e4m3"` | `str` / `list` | Target dtype(s) $[B, C, D, \dots]$ for each pass $k = 0 \dots K-1$. Accepts a single dtype repeated $K$ times (`"fp8_e4m3"`) or a comma-separated / YAML list of heterogeneous dtypes (`"fp8_e5m2,fp8_e4m3,fp4_e2m1"`). Supports `fp32`, `bf16`, `fp16`, `tf32`, `fp8_e4m3`, `fp8_e5m2`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1`, `fp4_e3m0`, `int8`, `int4`, `uint4`, `int2`, and arbitrary `eXmY` (e.g. `e5m3`) or `intB` (e.g. `int6`). |
| `emulate_multipass_lhs` | `True` | `bool` | Whether to apply multi-pass decomposition to the LHS (activation) operand. |
| `emulate_multipass_rhs` | `True` | `bool` | Whether to apply multi-pass decomposition to the RHS (weight) operand. |
| `emulate_lhs_src_dtype` | `""` | `str` | Optional override for the LHS (activation) source dtype $A_{\text{lhs}}$. |
| `emulate_lhs_num_passes` | `-1` | `int` | Optional override for the number of passes $K_A$ on the LHS (activation) operand. |
| `emulate_lhs_pass_dtypes` | `""` | `str` / `list` | Optional override for the LHS (activation) per-pass target dtypes $[B_0, B_1, \dots]$. |
| `emulate_lhs_scale_mode` | `""` | `str` | Optional override for the LHS (activation) residual scaling mode. |
| `emulate_rhs_src_dtype` | `""` | `str` | Optional override for the RHS (weight) source dtype $A_{\text{rhs}}$. |
| `emulate_rhs_num_passes` | `-1` | `int` | Optional override for the number of passes $K_B$ on the RHS (weight) operand. |
| `emulate_rhs_pass_dtypes` | `""` | `str` / `list` | Optional override for the RHS (weight) per-pass target dtypes $[C_0, C_1, \dots]$. |
| `emulate_rhs_scale_mode` | `""` | `str` | Optional override for the RHS (weight) residual scaling mode. |
| `emulate_scale_mode` | `"per_tensor_pow2"` | `str` | Per-pass residual scaling strategy: `"per_tensor_pow2"` (power-of-two tensor scale), `"per_tensor_amax"` (FP32 tensor amax scale), `"per_channel_pow2"` (contracting-axis channel power-of-two scale), `"per_channel_amax"`, `"per_block_pow2"` (microscaling block power-of-two scale), `"per_block_amax"`, `"bit_shift"` (exact integer bit-shift decomposition), or `"none"` (unscaled direct rounding). |
| `emulate_block_size` | `128` | `int` | Block size along the contracting dimension when `emulate_scale_mode` is `"per_block_pow2"` or `"per_block_amax"`. |
| `emulate_rounding_mode` | `"rne"` | `str` | Rounding mode applied when quantizing each pass: `"rne"` (round-to-nearest-even), `"stochastic"` (unbiased stochastic rounding), or `"truncate"` (round toward zero). |
| `emulate_flush_subnormals` | `False` | `bool` | If `True`, flushes values smaller than the minimum normal of the pass dtype to zero. |
| `emulate_cross_term_mode` | `"triangular"` | `str` | Cross-term pass pair selection policy: `"triangular"` ($i + j \le \text{max\_pass\_degree}$, e.g. `X3` for $K=2$), `"full"` (all $K_A \times K_B$ pairs, e.g. `X4`), `"asymmetric_lhs"` (`X2` where RHS is 1 pass), `"asymmetric_rhs"` (`X2` where LHS is 1 pass), `"karatsuba"` (3-pass Karatsuba recombination), or `"custom"` (explicit pairs via `emulate_active_passes`). |
| `emulate_max_pass_degree` | `-1` | `int` | Maximum pass index sum $i + j$ included when `emulate_cross_term_mode="triangular"`. `-1` defaults to $\max(K_A - 1, K_B - 1)$. |
| `emulate_active_passes` | `""` | `str` / `list` | Explicit comma-separated `i:j` pass index pairs when `emulate_cross_term_mode="custom"` (e.g., `"0:0,0:1,1:0"`). |
| `emulate_acc_dtype` | `"fp32"` | `str` | Accumulator dtype used for each pass's dot product and cross-pass summation (`"fp32"`, `"bf16"`, `"int32"`, or custom `"eXmY"`). |
| `emulate_out_dtype` | `"auto"` | `str` | Output dtype cast applied after accumulation (`"auto"` preserves the original matmul output dtype, or specify `"bf16"`, `"fp32"`, etc.). |
| `emulate_bwd_mode` | `"multipass_bwd"` | `str` | Backward pass mode: `"multipass_bwd"` (custom VJP supporting independent multi-pass emulation on `fwd`, `dlhs`, and `drhs`) or `"ste"` (Straight-Through Estimator). |
| `emulate_fwd` | `True` | `bool` | Whether to apply multi-pass emulation on the forward pass GEMM (`fprop`). |
| `emulate_dlhs` | `False` | `bool` | Whether to apply multi-pass emulation on the backward activation-gradient GEMM (`dgrad`: $dX = dY \cdot W^T$) when `emulate_bwd_mode="multipass_bwd"`. |
| `emulate_drhs` | `False` | `bool` | Whether to apply multi-pass emulation on the backward weight-gradient GEMM (`wgrad`: $dW = X^T \cdot dY$) when `emulate_bwd_mode="multipass_bwd"`. |
| `emulate_dlhs_cross_term_mode` | `""` | `str` | Optional cross-term mode override for the backward `dlhs` GEMM. |
| `emulate_drhs_cross_term_mode` | `""` | `str` | Optional cross-term mode override for the backward `drhs` GEMM. |
| `emulate_target_preset` | `"all_linears"` | `str` | Module selector preset: `"all_linears"`, `"mlp"`, `"attn"`, `"qkv"`, `"attn_out"`, `"moe"`, `"moe_router"`, or `"all_including_attn_scores"`. |
| `emulate_module_path` | `".*"` | `str` | Regex pattern matched against normalized module paths to select target layers. |
| `emulate_exclude_path` | `"(^\|/)(token_embedder\|...)"` | `str` | Regex pattern of module paths to exclude from multi-pass emulation. |
| `emulate_layers` | `""` | `str` / `list` | Optional decoder layer index filter (e.g., `"14"`, `"0,1,2"`, or `"8-15"`). Works with both `scan_layers=True` and `scan_layers=False`. |
| `emulate_exclude_layers` | `""` | `str` / `list` | Optional decoder layer indices to exclude from multi-pass emulation. |
| `emulate_op_names` | `["dot_general", "gmm", "ragged_dot"]` | `list` / `str` | Intercepted operation types (`dot_general`, `einsum`, `gmm`, `ragged_dot`). |
| `emulate_start_step` | `0` | `int` | Training step (inclusive) at which multi-pass emulation activates. |
| `emulate_end_step` | `-1` | `int` | Training step (inclusive) after which multi-pass emulation deactivates (`-1` = never). |
| `emulate_reference_model` | `False` | `bool` | If `True`, also applies multi-pass emulation to the Reference model in RL/DPO (default `False`). |
| `emulate_teacher_model` | `False` | `bool` | If `True`, also applies multi-pass emulation to the Teacher model in Distillation (default `False`). |
| `emulate_bake_weight_multipass_on_save` | `False` | `bool` | If `True`, bakes static multi-pass weight reconstruction (`sum_k p_k`) into saved checkpoints. |
| `emulate_apply_weight_multipass_on_load` | `False` | `bool` | If `True`, applies static multi-pass weight reconstruction when loading checkpoints. |
| `emulate_log_diagnostics` | `True` | `bool` | If `True`, logs matched module paths and active multi-pass configuration summaries. |
| `emulate_rules` | `[]` | `list` | Optional YAML list of granular per-module/per-layer `MultipassRuleConfig` dictionaries. |

---

## 4. Usage Examples

### A. SFT with RFC `F8E4M3_F8E4M3_F32_X3` and 128-Block Microscaling
```bash
python3 codesign/train_sft.py codesign/multi_pass/configs/sft_multipass.yml \
  run_name=qwen3_sft_fp8_x3 \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \
  emulate_scale_mode=per_block_pow2 \
  emulate_block_size=128
```

### B. Custom Heterogeneous Pass Dtypes ($A \to [B, C, D]$)
Decompose `bf16` activations and weights into 3 passes of heterogeneous dtypes `[fp8_e5m2, fp8_e4m3, fp4_e2m1]` with triangular cross-terms:
```bash
python3 codesign/train_sft.py codesign/multi_pass/configs/sft_multipass.yml \
  run_name=qwen3_sft_heterogeneous_3pass \
  emulate_multipass_enabled=True \
  emulate_src_dtype=bf16 \
  emulate_pass_dtypes="fp8_e5m2,fp8_e4m3,fp4_e2m1" \
  emulate_scale_mode=per_block_pow2 \
  emulate_cross_term_mode=triangular \
  emulate_acc_dtype=fp32
```

### C. Targeted Layer & Mid-Training Step Schedule (`scan_layers=True` or `False`)
Apply 3-pass `F8E4M3_F8E4M3_F32_X3` only to decoder layer 14 starting at step 25:
```bash
python3 codesign/train_sft.py codesign/multi_pass/configs/sft_multipass.yml \
  run_name=qwen3_sft_l14_step25 \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \
  emulate_layers="14" \
  emulate_start_step=25
```

### D. Pre-Training, RL (GRPO), DPO, Distillation, and Evaluation
```bash
# Pre-Training
python3 codesign/train.py src/maxtext/configs/base.yml \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3

# GRPO / RL Post-Training (Actor emulated, Reference model kept full precision)
python3 codesign/train_rl.py codesign/multi_pass/configs/rl_multipass.yml \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3

# Knowledge Distillation (Student emulated, Teacher kept full precision)
python3 codesign/train_distill.py codesign/multi_pass/configs/distill_multipass.yml \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3

# Evaluation
python3 codesign/eval.py codesign/multi_pass/configs/eval_multipass.yml \
  load_parameters_path=/path/to/checkpoint \
  emulate_multipass_enabled=True \
  emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \
  --native_eval
```
