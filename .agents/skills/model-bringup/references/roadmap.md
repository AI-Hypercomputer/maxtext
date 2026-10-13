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
| **6** | Write Layer Tests | `tests/unit/{model_name}_layers_test.py` | CPU verification (`assert_allclose`) against PyTorch reference | Record test results & numerical tolerance |
| **7** | Define Param Mapping | `src/maxtext/checkpoint_conversion/utils/param_mapping.py` | Unit tests for transposition and reshape hooks | Record mapping logic & hook details |
| **8** | Convert Mini Checkpoint | Run `to_maxtext` on TPU VM using mini safetensors subset | Valid Orbax checkpoint output in `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax` | Record exact conversion CLI command |
| **9** | Mini E2E Logits & Decode Check | Run `forward_pass_logit_checker` and 16-token autoregressive decode | Logits KL Div $\\le 10^{-3}$ and exact token-for-token matching gibberish output | Record verification metrics and decode outputs |
| **10** | Full Checkpoint & Verification | Run full `to_maxtext`, full logits check, and full decode | Full model logits match & meaningful decode | Record full run commands & output samples |
| **11** | Handover Wrap-up | `{model_name}_handover.md` in MaxText repository root | Complete summary of pitfalls, gotchas, and full CLI commands | Mark all phases `DONE` |

---

## 2. Handover Document Template (`{model_name}_handover.md`)

Maintain `{model_name}_handover.md` under the MaxText repository root throughout bring-up.

### Required Structure:
```markdown
# {Model Name} Bring-up Handover Document

## 1. HF Model Architectural Findings
- **HuggingFace Model ID**: e.g., `THUDM/glm-4-9b-chat`
- **Decoder Pattern**: (e.g., Homogeneous repeated / Dense first 3 layers + MoE / Local-Global Attention cycle)
- **Minimal Subset Layer Count ($N$)**: e.g., 1 layer, 4 layers
- **Key Parameters**: `vocab_size`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `rms_norm_eps`, `rope_theta`, etc.

## 2. Phase-by-Phase Status & Phase 4 Verification Checklist
- [x] Phase 1: HF Discovery & Mini Safetensors Creation (DONE)
- [x] Phase 2: JAX Layer Implementation & Layer Tests (DONE)
- [x] Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion (DONE)
- [x] Phase 4: Comprehensive Mini-Model Verification, E2E Logits & 16-Token Decode (DONE)
  - [x] **Gate 0 (Slicing & Conversion Parity)**: `slice_model.py` + `to_huggingface.py` round-trip (`max_abs_diff == 0.0`)
  - [x] **Dimension 1 (Challenging Inputs)**: Sequence/chunk boundaries, padded batches (+ attribution probes), packed documents (`segment_ids`), special/boundary tokens, and multimodal/3D MRoPE grids
  - [x] **Dimension 2 (Forward Boundaries)**: Component outputs (ViT, merger, post-scatter embeddings, linear/full attention sub-stages, MLP/MoE, MTP sub-stages) & complete mini-model logits
  - [x] **Dimension 3 (MoE / Dense Coverage)**: Deliberate/forced routing & load-balancing loss (MoE) or SwiGLU sub-activations & zero routing overhead (Dense)
  - [x] **Dimension 4 (Enabled Heads & Objectives)**: Pre-aggregation loss components exposed; 4-way ablation (`Config_A` Main CE, `Config_B` +Z-loss, `Config_C` MTP only, `Config_D` Combined)
  - [x] **Dimension 5 (Backward Sensitivity & Independence)**: Per-parameter-group E2E gradients + isolated block-level upstream VJP probes (`dX`, `dW`)
  - [x] **Dimension 6 (Multi-Step Stateful Updates & Freezing)**: 3-step stateful AdamW with changing batches (weights $W^{(3)}$ + moments $m^{(3)}, v^{(3)}$) & frozen parameter invariance
  - [x] **Supplemental Check S.1 (`scan_layers=True` Parity)**: Scanned `param_mapping`, forward logits, loss, and backward gradients vs. `scan_layers=False`
  - [x] **Supplemental Check S.2 (Asymmetric 3D MRoPE & Negative Control)**: Non-square `[T, H, W]` grid + axis-swap negative control
  - [x] **Supplemental Check S.3 (Multi-Chip 2D Sharding)**: Physical `PartitionSpec`s + `fsdp=2, tensor=2` forward/backward parity vs. unsharded
  - [x] **Supplemental Check S.4 (`bfloat16` Mixed Precision)**: Zero `NaN`/`Inf`, `100%` top-1 logit agreement, and `cosine_sim > 0.998` across all gradient groups
  - [x] **CLI E2E Logits & 16-Token Autoregressive Decode**: `forward_pass_logit_checker` (KL Div $\le 10^{-3}$) & exact 16-token greedy decode match

## 3. Bugs Encountered & Solutions
- **Phase 2 (JAX Layers)**: [Description of bug, root cause, and how it was fixed]
- **Phase 3 (Checkpoint Conversion)**: [Transposition mismatch, key naming issue, etc.]
- **Phase 4 (Mini Verification, Logits & Decode)**: [Multimodal scatter token ID, MTP projection matmul_precision, reverse QKV fusion hooks, RoPE frequency mismatch, KV cache indexing, etc.]

## 4. Pitfalls & Model-Specific Gotchas (Session Wrap-up)
- Summary of unexpected model behaviors, unusual head projections, bias handling, tied embeddings, etc.

## 5. Full Executable Reproduction Commands
- **Mini Safetensors Slicing**:
  `python3 slice_hf_model.py ...`
- **Mini Checkpoint Conversion**:
  `python3 -m maxtext.checkpoint_conversion.to_maxtext ...`
- **6-Dimension Mini-Model Verification & 4 Supplemental Checks**:
  `python3 -u tests/integration/{model_name}_mini_verification_test.py`
  `python3 -u tests/integration/{model_name}_mini_supplemental_test.py`
- **Forward Pass Logits Check**:
  `python3 -m tests.utils.forward_pass_logit_checker ...`
- **Autoregressive Decode Check**:
  `python3 -m maxtext.inference.decode ...`
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
