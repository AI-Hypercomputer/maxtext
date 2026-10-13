# Implementation Roadmap & Hardcoded Parameters Reference

This guide provides an end-to-end implementation roadmap for onboarding a new model to MaxText, defines the structure of the required handover document, and documents the origin/source for every hardcoded configuration and weight translation value.

---

## 1. Implementation Roadmap Checklist

Follow this execution order when onboarding a new model:

| Step | Action | Files Modified / Created | Target Validation | Handover Update |
| :--- | :--- | :--- | :--- | :--- |
| **1** | HF Discovery & Pattern Analysis | `None` (Python inspection) | Identify decoder pattern (homogeneous vs inhomogeneous) and determine minimal subset $N$ layers | Initialize `{model_name}_handover.md` with HF findings |
| **2** | Build Mini HF Model | Mini HF model in `/dev/shm/hf_mini/{model_name}_mini` (approach is the agent's choice; see `verification.md` §1.1) | Valid reduced config, tokenizer, and weights covering every distinct component | Note approach, path & sizes in handover doc |
| **3** | Add Model Config | `src/maxtext/configs/models/{model_name}.yml` | Config load check | Log model dimensions in handover doc |
| **4** | Register Model ID | `src/maxtext/utils/globals.py` & `src/maxtext/checkpoint_conversion/utils/hf_model_configs.py` | Import resolution | Update status to `IN PROGRESS` |
| **5** | Code JAX Layers | `src/maxtext/layers/decoders.py` (or new custom layer files) | Python syntax, shapes, and autoregressive decode support | Document custom layer design & bugs found |
| **6** | Write Layer Tests | `tests/unit/{model_name}_layers_test.py` | CPU verification (`assert_allclose`) against PyTorch reference | Record test results & numerical tolerance |
| **7** | Define Param Mapping | `src/maxtext/checkpoint_conversion/utils/param_mapping.py` & `src/maxtext/checkpoint_conversion/utils/hf_shape.py` | Unit tests for transposition, reshape hooks, and target shapes | Record mapping logic & hook details |
| **8** | Convert Mini Checkpoint | Run `to_maxtext` (or `to_huggingface`) on TPU VM using the mini model | Valid Orbax checkpoint (`$MINI_CKPT`, e.g. `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax/0/items`) | Record exact conversion CLI command |
| **9** | Mini-Model Verification, E2E Logits & Decode Check | Gate 0 parity, 6-dimension mini verification + supplemental checks (`tests/integration/{model_name}_mini_*_test.py`), `forward_pass_logit_checker`, and 16-token autoregressive decode | All applicable dimensions within thresholds (see `verification.md` §1.3), Logits KL Div $\le 10^{-3}$, and exact token-for-token matching gibberish output | Fill in the Phase 4 checklist with metrics and decode outputs |
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
<!-- Status: NOT STARTED / IN PROGRESS / BLOCKED / DONE. Mark non-applicable items `N/A` with a reason. -->
- [ ] Phase 1: HF Discovery & Mini Model Plan (`<status>`)
- [ ] Phase 2: JAX Layer Implementation & Layer Tests (`<status>`)
- [ ] Phase 3: Checkpoint Mapping & Mini-Checkpoint Conversion (`<status>`)
- [ ] Phase 4: Mini-Model Verification, E2E Logits & 16-Token Decode (`<status>`)
  - Paste and fill in the full Phase 4 checklist template from `.agents/skills/model-bringup/references/verification.md` (§7).

## 3. Bugs Encountered & Solutions
- **Phase 2 (JAX Layers)**: [Description of bug, root cause, and how it was fixed]
- **Phase 3 (Checkpoint Conversion)**: [Transposition mismatch, key naming issue, etc.]
- **Phase 4 (Mini Verification, Logits & Decode)**: [Multimodal placeholder token ID, missing matmul_precision on a projection, fused-weight mapping hooks, RoPE frequency mismatch, KV cache indexing, etc.]

## 4. Pitfalls & Model-Specific Gotchas (Session Wrap-up)
- Summary of unexpected model behaviors, unusual head projections, bias handling, tied embeddings, etc.

## 5. Full Executable Reproduction Commands
- **Mini Model Construction** (approach chosen in Phase 4 Gate 0, and why):
  `<commands or script used>`
- **Mini Checkpoint Conversion**:
  `python3 -m maxtext.checkpoint_conversion.to_maxtext ...` (or `to_huggingface` for the reverse direction)
- **6-Dimension Mini-Model Verification & Supplemental Checks**:
  `python3 -u tests/integration/{model_name}_mini_verification_test.py`
  `python3 -u tests/integration/{model_name}_mini_supplemental_test.py`
- **Forward Pass Logits Check**:
  `python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml model_name={model_name} ... --golden_logits_path=... --max_kl_div=1e-3`
- **Autoregressive Decode Check**:
  `python3 -m maxtext.inference.decode src/maxtext/configs/base.yml model_name={model_name} ...`
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
| `base_mlp_dim` | Hidden dimension of the MLP block (and `base_moe_mlp_dim` for MoE) | `intermediate_size` (and `moe_intermediate_size`) |
| `logits_via_embedding` | Whether output LM head shares token embedding weights | `tie_word_embeddings` |
| `decoder_block` | Custom block name identifier | Custom name string (e.g. `"qwen3"`, `"llama3"`, `"gemma3"`) |

### B. Normalization Parameters
*   **Epsilon (`normalization_layer_epsilon`)**:
    *   *Where to set*: Add `normalization_layer_epsilon` to `{model_name}.yml` (or pass to layer initialization).
    *   *Source*: `config.json` $\rightarrow$ `rms_norm_eps` (or `layer_norm_eps`).
    *   *Gotcha*: If this mismatches even slightly (e.g., $10^{-6}$ vs $10^{-5}$), RMSNorm output will diverge.

### C. Rotary Embeddings (RoPE) Config
*   **RoPE Base Frequency (`rope_max_timescale`)**:
    *   *Where to set*: Add to `{model_name}.yml` under `rope_max_timescale` (and `local_rope_max_timescale` if local/global attention uses a different theta).
    *   *Source*: `config.json` $\rightarrow$ `rope_theta` (or `rope_parameters` / `rope_scaling` settings).
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
