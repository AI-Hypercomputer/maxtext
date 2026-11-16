# Implementation Roadmap & Hardcoded Parameters Reference

This guide provides an end-to-end implementation roadmap for onboarding a new model to MaxText, defines the structure of the required handover document, and documents the origin/source for every hardcoded configuration and weight translation value.

---

## 1. Implementation Roadmap Checklist

Follow this execution order when onboarding a new model:

| Step | Action | Files Modified / Created | Target Validation | Handover Update |
| :--- | :--- | :--- | :--- | :--- |
| **1** | HF Discovery & Pattern Analysis | `None` (Python inspection) | Identify decoder pattern (homogeneous vs inhomogeneous) and determine minimal subset $N$ layers | Initialize `{model_name}_handover.md` with HF findings |
| **2** | Create Mini HF Safetensors | Sliced safetensors in `/dev/shm/hf_mini/{model_name}_{N}layers` | Valid minimal config, tokenizer, embeddings, $N$ layers, and `lm_head` | Note path & layer count in handover doc |
| **3** | Add Model Config | `src/maxtext/configs/models/{model_name}.yml` | Config load check | Log model dimensions in handover doc |
| **4** | Register Model ID | `src/maxtext/utils/globals.py` & `src/maxtext/checkpoint_conversion/utils/hf_model_configs.py` | Import resolution | Update status to `IN PROGRESS` |
| **5** | Code JAX Layers | `src/maxtext/layers/decoders.py` (or new custom layer files) | Python syntax, shapes, and autoregressive decode support | Document custom layer design & bugs found |
| **6** | Write Layer Tests | `tests/unit/{model_name}_vs_reference_test.py` | CPU verification (`assert_allclose`) against PyTorch reference | Record test results & numerical tolerance |
| **7** | Define Param Mapping | `src/maxtext/checkpoint_conversion/utils/param_mapping.py` | Unit tests for transposition and reshape hooks | Record mapping logic & hook details |
| **8** | Convert Mini Checkpoint | Run `to_maxtext` on TPU VM using mini safetensors subset | Valid Orbax checkpoint output in `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax` | Record exact conversion CLI command |
| **9** | Mini-Model-Verification | Run [mini-model-verification](mini_verification.md) on trimmed/mini checkpoints | Trimmed/mini model parity against PyTorch reference | Record mini-model-verification metrics |
| **10** | Full Checkpoint & Verification | Convert full `scanned` & `unscanned` checkpoints; run multi-token masked prefill ([logits_checker.md](logits_checker.md)) & training/SFT checks | `scanned` & `unscanned` masked prefill parity and training stability | Record full run commands & metrics |
| **11** | FLOPs, Benchmark & Roofline | `src/maxtext/utils/maxtext_utils.py` & [performance_and_roofline.md](performance_and_roofline.md) | Unit-tested TFLOPs, TPU training benchmark (XPlane + HLO HBM), and analytical vs measured roofline report | Record benchmark & roofline summary |
| **12** | Handover Wrap-up | `{model_name}_handover.md` in MaxText repository root | Complete summary of pitfalls, gotchas, and full CLI commands | Mark all phases `DONE` |

---

## 2. Handover Document Template (`{model_name}_handover.md`)

Maintain `{model_name}_handover.md` under the MaxText repository root so architectural findings, quantitative verification results, exact reproduction commands, and log/artifact paths are auditable in one place:

```markdown
# {Model Name} Bring-up Handover Document

## 1. HF Model Architectural Findings & Configs
- **Sources**: HF model card (`{hf_model_id}`), PyTorch reference (`modeling_{model_name}.py`), MaxText branch/commit.
- **Architecture Highlights**: Total/active params, layer types & `scan_layers=true` structure, attention (Q/KV heads, QK-norm, gating, sliding window, RoPE/NoPE), MLP/MoE (router, bottleneck, experts/top-k, shared experts), norms, residual gates, and embedding/head/soft-cap.

| Parameter | Value |
| :--- | :--- |
| Total / Active Parameters | ... |
| Vocab / Hidden / Layers (`vocab_size` / `base_emb_dim` / `base_num_decoder_layers`) | ... |
| Layer Pattern & Scanned Layout | ... |
| Attention Specs (Q/KV heads, `head_dim`, window, RoPE) | ... |
| MLP / MoE Specs (`num_experts`, `num_experts_per_tok`, dims, router) | ... |
| Norms / Residual / Head Specs | ... |

## 2. Verification & Evidence Summary
| Task | Status | Log / Artifact | Key Result Notes |
| :--- | :--- | :--- | :--- |
| 1. Model, Config & Unit Tests | DONE / IN PROGRESS | Branch / test log | e.g., N/N CPU unit tests pass |
| 2. Checkpoint Conversion (`scanned` & `unscanned`) | ... | Conversion logs & ckpt paths | `scan_layers=true` & `false` Orbax ckpts |
| 3. Multi-Token Masked Prefill (`scanned` & `unscanned`) | ... | JSON reports & console log | Cosine sim, max abs, KL, ΔCE |
| 4. Mini-Model-Verification vs PyTorch | ... | Tier reports & log | X/Y checks pass across tiers |
| 5. Training & SFT Correctness | ... | `scanned` / `unscanned` logs | Pretrain & SFT loss ranges, HBM |
| 6. Pre-Training Benchmark & Roofline | ... | Bench log, HLO memory, XPlane, Roofline doc | Step time, TFLOP/s (MFU), HBM vs roofline |

## 3. Working Log & Verification Evidence

### 3.1 Code Changes & Unit Tests
- **Result**: Unit tests passed (`N/N`).
- **Command**: `python3 -m pytest ...`
- **Logs / Branch**: ...

### 3.2 Checkpoint Conversion (`scanned` & `unscanned`)
- **Commands**: Exact `to_maxtext` CLI commands for `scan_layers=true` and `scan_layers=false`.
- **Checkpoint, Golden & Log Paths**: Source HF path, `scanned`/`unscanned` (and resharded) Orbax paths, reference golden `.npz`/`.jsonl` paths, and conversion logs.

### 3.3 Multi-Token Masked Prefill (`scanned` & `unscanned`)
- **Results Table**: `Checkpoint | Mode (Router Replay / Unforced) | Reference | Mean/Min Cosine Sim | Mean/Max Abs Err | Mean KL | ΔCE | Verdict`
- **Commands**: Exact `zero_gen_logits_checker` / `forward_pass_logit_checker` commands.
- **Evidence Logs**: JSON report paths and console logs.

### 3.4 Mini-Model-Verification vs PyTorch
- **Setup & Tier Results**: Trimmed/sliced tiers tested and pass counts per tier + coverage summary.
- **Command & Logs**: Exact test runner command and report/log links.

### 3.5 Training & SFT Correctness (`scanned` & `unscanned`)
- **Results Table**: `Mode | Pretrain loss (step 0 -> K) | SFT loss (step 0 -> K) | Step time | Compiled HBM`
- **Commands & Logs**: Exact `pre_train.train` and `train_sft_native` commands and console logs.

### 3.6 Pre-Training Benchmark & Roofline Analysis
- **Results Table**: Topology, precision, seq len, batch tokens, `(FSDP, TP, EP)`, step TFLOPs, steady-state `s/step`, `TFLOP/s/device` (MFU %), `Tokens/s/device`, compiled HBM vs roofline lower bound.
- **Command & Artifacts**: Benchmark command (including `LIBTPU_INIT_ARGS`), console log, HLO memory report, XPlane profile paths, and roofline report link.

## 4. Bugs & Model-Specific Pitfalls
- Concise list of bugs encountered, root causes, fixes, and architectural gotchas.
```

---

## 3. Hardcoded Values & Configs Source Registry

When bringing up a model, multiple hyperparameters and mapping patterns must be hardcoded. Use the registry below to locate their exact sources:

### A. Model Architecture Dimensions (`{model_name}.yml`)
All model parameters added to the `.yml` config file must correspond exactly to their HuggingFace counterparts:

| MaxText Key | Description | HuggingFace Config Source Key |
| :--- | :--- | :--- |
| `vocab_size` | Size of the token vocabulary | `vocab_size` |
| `base_emb_dim` | Model hidden state dimension ($d_{model}$) | `hidden_size` |
| `base_num_decoder_layers` | Number of transformer layers | `num_hidden_layers` |
| `base_num_query_heads` | Attention query heads | `num_attention_heads` |
| `base_num_kv_heads` | Key/Value attention heads | `num_key_value_heads` |
| `head_dim` | Dimension per head | `hidden_size // num_attention_heads` (or `head_dim`) |
| `mlp_dim` | Hidden dimension of the MLP block | `intermediate_size` |
| `decoder_block` | Custom block name identifier | Custom name string (e.g. `"qwen2"`, `"llama"`, `"glm"`) |

### B. Normalization Parameters
*   **Epsilon (`rms_norm_eps` or `layer_norm_epsilon`)**:
    *   *Where to set*: Add to `{model_name}.yml` or hardcode in layer initialization.
    *   *Source*: `config.json` $\\rightarrow$ `rms_norm_eps` (or `layer_norm_eps`).
    *   *Gotcha*: If this mismatches even slightly (e.g., $10^{-6}$ vs $10^{-5}$), RMSNorm output will diverge.

### C. Rotary Embeddings (RoPE) Config
*   **RoPE Base Frequency (`rope_theta`)**:
    *   *Where to set*: Add to `{model_name}.yml` under `rope_base` or `rope_theta`.
    *   *Source*: `config.json` $\\rightarrow$ `rope_theta` (or `rope_embedding_base` / `rope_scaling` settings).
*   **RoPE Dimensions**:
    *   *Gotcha*: Some models apply RoPE only to a fraction of the head dimension (e.g., `partial_rotary_factor`). Look for `rotary_dim` in `modeling_{model_name}.py`.

### D. Parameter Naming Translation Map (`param_mapping.py`)
Keys and paths used in `PARAM_MAPPING` must be extracted from the parameters naming conventions:

*   **HF Parameter Naming**:
    *   *Source*: Inspect safetensors metadata keys, or load PyTorch model weights state dict keys: `model.state_dict().keys()`.
*   **MaxText Parameter Naming**:
    *   *Source*: Print JAX variables structure after initializing your custom JAX model:
        ```python
        variables = model.init(rng, inputs)
        print(variables['params'].keys())
        ```

### E. Gated MLP/SwiGLU Settings
*   **SwiGLU intermediate dimension**:
    *   *Gotcha*: For SwiGLU, MaxText implements the gate and up projections. Ensure your layer dimensions and kernel shapes match the `intermediate_size` from `config.json` and are mapped properly to the JAX keys `wi_0` and `wi_1`.
