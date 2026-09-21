# Description

Under FP8 training (`quantization='fp8_full'`), the output vocabulary projection (`logits_dense`) previously remained in full precision (BF16).

This PR introduces `quantize_logits_proj=true` to quantize the logits projection via Qwix rule interception (`decoder/logits_dense.*`), with an optional `logits_proj_quant_calibration_method` override. For architectures with Multi-Token Prediction (MTP) such as DeepSeek-V3, MTP modules reuse `decoder/logits_dense`, automatically extending FP8 quantization across all projection passes.

BUGS: b/559196820

## Changes
* **Config & Validation:** `src/maxtext/configs/base.yml`, `src/maxtext/configs/types.py`
  - Added `quantize_logits_proj` (default: `false`) and `logits_proj_quant_calibration_method` (default: `""`).
  - Added validations requiring `quantization='fp8_full'`, `logits_via_embedding=False` (untied embeddings), `num_vocab_tiling=1`, and `logits_dot_in_fp32=False`.
* **Qwix Interception:** `src/maxtext/layers/quantizations.py`
  - Prepended a dedicated `QtRule(module_path="decoder/logits_dense.*", ...)` targeting `dot_general` ahead of general layer rules.
  - Inherits the global calibration method by default or applies the specified override.
* **Documentation:** `docs/reference/core_concepts/quantization.md`
  - Documented flag behavior, calibration inheritance, architectural constraints, and MTP head reuse.
* **Testing:** `tests/unit/quantizations_test.py`
  - Added `LogitsProjQwixTest` verifying rule interception and calibration inheritance/override under CPU abstract evaluation.

## Details

1. **Compute & Speedup:**
   - In large-vocabulary models like DeepSeek-V3 ($V = 129{,}280, d_{\text{model}} = 7{,}168$), the output head is a compute-heavy GEMM (~0.93B parameters, $2 \cdot d_{\text{model}} \cdot V$ FLOPs/token).
   - In DeepSeek-V3, the vocabulary projection accounts for ~2.5% of training FLOPs without MTP ($M=0$) and ~5.0% with MTP ($M=1$), yielding a theoretical roofline of ~1.2% to ~2.4% end-to-end speedup in FP8 (empirical gains are typically at most 1–2% due to unquantized softmax and communication overhead).
   - MTP reuses `decoder/logits_dense` via `apply_output_head`, accelerating both main and MTP projection passes ($(1 + M)$ invocations).

2. **Stability & Default Policy:**
   - The DeepSeek-V3 paper leaves the output head unquantized to safeguard cross-entropy convergence.
   - Other frameworks (such as Megatron-LM) support quantizing the output projection (e.g. to block-scaled MXFP8 on GB300) when training stability permits.
   - `quantize_logits_proj` defaults to `false` to maintain baseline convergence by default while providing an opt-in knob for throughput optimization.


# Tests

### Unit Tests
Verified Qwix rule generation and interception under CPU abstract evaluation:
```bash
JAX_PLATFORMS=cpu pytest tests/unit/quantizations_test.py -k "LogitsProjQwixTest"
```

### Rule Interception Tracing
Verified that Qwix intercepts `decoder/logits_dense` for both the main model (`dot_general0`) and MTP (`dot_general1`):
```
# quantize_logits_proj=true
[QWIX] module='decoder/logits_dense' op=dot_general0 rule=0
[QWIX] module='decoder/logits_dense' op=dot_general1 rule=0

# quantize_logits_proj=false
[QWIX] module='decoder/logits_dense' op=dot_general0 rule=None
[QWIX] module='decoder/logits_dense' op=dot_general1 rule=None
```

### End-to-end Test

deepseek3-671b: b/559196820

# Checklist

Before submitting this PR, please make sure (put X in square brackets):
- [x] I have performed a self-review of my code. For an optional AI review, add the `gemini-review` label.
- [x] I have necessary comments in my code, particularly in hard-to-understand areas.
- [x] I have run end-to-end tests and provided workload links above if applicable.
- [x] I have made or will make corresponding changes to the doc if needed, including adding new documentation pages to the relevant Table of Contents (toctree directive) as explained in [our documentation](https://maxtext.readthedocs.io/en/latest/development.html#adding-new-documentation-files).
