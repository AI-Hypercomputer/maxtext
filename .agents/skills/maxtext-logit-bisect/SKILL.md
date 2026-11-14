---
name: maxtext-logit-bisect
description: >-
  Localizes root causes of MaxText forward-pass logit and activation
  divergences against HuggingFace PyTorch reference models. Use when
  forward_pass_logit_checker (or Phase 4 of model-bringup) fails or reports
  elevated KL divergence to audit 1:1 HuggingFace-to-MaxText config mappings,
  classify divergence signatures, bisect layerwise activation dumps across any
  model architecture, and optionally run single-layer FP64 weight recomputation.
---

# MaxText Forward-Pass Logit & Activation Bisection

When `forward_pass_logit_checker.py` (see
[model-bringup logits_checker.md](../model-bringup/references/logits_checker.md))
reports high KL divergence or top-1 token mismatches, localize the first
diverging component systematically instead of guessing hyperparameters.

| Script | Purpose |
| :--- | :--- |
| [`scripts/hf_cfg_audit.py`](scripts/hf_cfg_audit.py) | Deterministic 1:1 audit of HuggingFace `config.json` vs. MaxText model `.yml` |
| [`scripts/bisect_layer_divergence.py`](scripts/bisect_layer_divergence.py) | Layerwise activation-dump bisection (any architecture) + optional FP64 single-layer weight check |
| [`scripts/bisect_utils.py`](scripts/bisect_utils.py) | FP64 math helpers, scanned/unscanned Orbax Zarr3 loader, and pluggable `LayerAdapter` |

---

## Step 1: Audit 1:1 Config Parity (`hf_cfg_audit.py`)

Verify that every architecture parameter in HuggingFace `config.json` matches
the MaxText model YAML (`src/maxtext/configs/models/<model>.yml`). Even small
mismatches in `rope_theta`, `rms_norm_eps`, MoE routing scales, or soft-capping
thresholds cause large multi-layer drift.

```bash
python3 .agents/skills/maxtext-logit-bisect/scripts/hf_cfg_audit.py \
  --hf-config /path/to/hf_checkpoint/config.json \
  --maxtext-yml src/maxtext/configs/models/<model>.yml
```

- Automatically unwraps `text_config` for multimodal checkpoints.
- Audits standard Transformer, MoE, MLA, RoPE/YaRN/Llama3 scaling, and
  Gemma-style soft-capping keys, plus any identically named keys present in both
  configs (extensible via `--key-map custom_map.json`).

---

## Step 2: Classify the Divergence Signature

Inspect per-position KL divergence `KL(P_HF || P_MaxText)` across token
positions `0 .. S-1`:

| Signature | Meaning | Primary Suspects |
| :--- | :--- | :--- |
| **`KL[0..S-1]` all huge (`> 1.0`)** | Input or vocabulary mismatch before layer 0 | BOS/EOS token prepending, chat template, tokenizer vocab padding, or embedding table transpose/scale |
| **`KL[0]` elevated (`> 1e-3`)** | Position-independent defect (RoPE and causal mask are no-ops at `pos=0`) | Weight transpose/reshape in checkpoint conversion, RMSNorm $\epsilon$ or zero-centered `(1 + w)` scale, MLP/MoE gating or router scaling, logit soft-capping, tied vs. untied `lm_head` |
| **`KL[0] < 1e-5`, spikes at `pos > 0`** | Position-dependent defect | RoPE half-rotation vs. interleaved layout (`permute_to_match_maxtext_rope`), RoPE/YaRN frequency scaling, causal or sliding-window mask off-by-one, QK-norm ordering |
| **Gradual growth across depth** | Compounding numerical noise floor | BF16 matmul accumulation vs. FP32 reference, or top-$k$ expert/sparse tie-breaking flips near BF16 boundaries |

---

## Step 3: Layerwise Activation Bisection (`bisect_layer_divergence.py`)

### Mode A: Model-Agnostic Activation Bisection (Default)

Save per-layer inputs/outputs from PyTorch (`register_forward_hook`) and MaxText
into `.npz` archives (`L00_in`, `L00_out`, ...). Run `bisect_layer_divergence.py`
without `--ckpt-uri` to pinpoint the first diverging layer, sub-block, and token
position across **any** architecture (dense, MoE, MLA, Mamba/GDN, multimodal):

```bash
python3 .agents/skills/maxtext-logit-bisect/scripts/bisect_layer_divergence.py \
  --hf-acts /tmp/hf_layer_io.npz \
  --mt-acts /tmp/mt_layer_io.npz \
  --subkey-templates "L{layer:02d}_attn,L{layer:02d}_mlp"
```

The script reports input `rel_l2`, output `rel_l2`, cosine similarity, worst
token position (`worst_pos`), and amplification ratio (`out_rel_l2 / in_rel_l2`)
for each layer, flagging the first layer that exceeds `--rtol`.

### Mode B: Single-Layer FP64 Weight Isolation (Optional `--ckpt-uri`)

When you need to distinguish a checkpoint weight bug from BF16 runtime drift on
the first diverging layer, pass `--ckpt-uri` (supports both unscanned
`layers_{i}` and `--scanned` `scan_layers=True` Orbax checkpoints) and an
optional `--adapter module:ClassName`:

```bash
python3 .agents/skills/maxtext-logit-bisect/scripts/bisect_layer_divergence.py \
  --hf-acts /tmp/hf_layer_io.npz \
  --mt-acts /tmp/mt_layer_io.npz \
  --ckpt-uri /path/to/orbax_ckpt/0/items \
  --layers 0,1,2 --scanned
```

- **`isolated_rel_l2` small (`< 1e-4`)**: Converted weights and layer math match
  the reference; runtime divergence is upstream drift or BF16 accumulation.
- **`isolated_rel_l2` large (`> 1e-2`)**: Defect is inside that layer's
  converted weights (transpose, missing scale, wrong expert/head slice) or
  layer formula.
