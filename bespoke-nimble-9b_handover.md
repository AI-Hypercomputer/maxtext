# Bespoke-Nimble-9B (LoRA Adapter on Qwen3.5-9B) Bring-up Handover Document

## 1. HF Model & Adapter Architectural Findings
- **HuggingFace Adapter ID**: `bespokelabs/Bespoke-Nimble-9B`
- **Base Model ID**: `Qwen/Qwen3.5-9B` (revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`)
- **Task**: `schema_candidate_classification_v1` (structured parallel boolean & enum classification)
- **LoRA Configuration**:
  - `peft_type`: `LORA`
  - `r` (rank): 16
  - `lora_alpha`: 32
  - `scaling`: $\alpha / r = 32 / 16 = 2.0$
  - `target_modules`: `["o_proj", "down_proj", "k_proj", "gate_proj", "in_proj_b", "in_proj_z", "in_proj_a", "out_proj", "q_proj", "in_proj_qkv", "v_proj", "up_proj"]`
  - `total_adapter_parameters`: 496 tensor weights (248 pairs of `lora_A` and `lora_B`)
- **Layer Distribution Across 32 Layers**:
  - 24 Linear Attention (GatedDeltaNet) layers (Layers 0..2, 4..6, 8..10, 12..14, 16..18, 20..22, 24..26, 28..30):
    - `linear_attn.in_proj_a` (dim $32 \times 4096$)
    - `linear_attn.in_proj_b` (dim $32 \times 4096$)
    - `linear_attn.in_proj_qkv` (dim $8192 \times 4096$)
    - `linear_attn.in_proj_z` (dim $4096 \times 4096$)
    - `linear_attn.out_proj` (dim $4096 \times 4096$)
    - `mlp.gate_proj` (dim $12288 \times 4096$)
    - `mlp.up_proj` (dim $12288 \times 4096$)
    - `mlp.down_proj` (dim $4096 \times 12288$)
    - (8 target modules per layer $\times$ 24 layers = 192 modules)
  - 8 Full Attention layers (Layers 3, 7, 11, 15, 19, 23, 27, 31):
    - `self_attn.q_proj` (dim $4096 \times 4096$)
    - `self_attn.k_proj` (dim $512 \times 4096$)
    - `self_attn.v_proj` (dim $512 \times 4096$)
    - `self_attn.o_proj` (dim $4096 \times 4096$)
    - `mlp.gate_proj` (dim $12288 \times 4096$)
    - `mlp.up_proj` (dim $12288 \times 4096$)
    - `mlp.down_proj` (dim $4096 \times 12288$)
    - (7 target modules per layer $\times$ 8 layers = 56 modules)
  - Total: $192 + 56 = 248$ modules (100% accounted for and mapped 1-to-1).

## 2. Phase-by-Phase Status
- [x] **Phase 1: Adapter Discovery & Architecture Analysis (DONE)**
  - Inspected `adapter_config.json`, `adapter_model.safetensors`, and `parallel_schema.py`.
  - Audited weight shapes and target module coverage against Qwen3.5-9B architecture.
- [x] **Phase 2: Checkpoint Conversion Mapping Verification (DONE)**
  - Checked `to_maxtext.py` `_setup_merge_mode_getter` logic.
  - Verified 248/248 LoRA modules match MaxText `PARAM_MAPPING['qwen3.5-9b']` keys.
- [x] **Phase 3: Checkpoint Conversion & LoRA Merging (DONE)**
  - Verified 4-layer mini subset conversion locally on TPU VM: `/dev/shm/hengtaoguo/checkpoints/bespoke-nimble-9b-mini-unscanned`.
  - Ran full 32-layer conversion with LoRA merged into base weights and saved to GCS: `gs://hengtaoguo-maxtext-logs/checkpoints/bespoke-nimble-9b/unscanned/2026-09-23/0/items`.
- [x] **Phase 4: E2E Autoregressive Decode & Golden Candidate Verification (DONE)**
  - Successfully ran `decode.py` on TPU v5p2 with the converted checkpoint generating fluent text for `"Paris is the"`.
  - Executed candidate scoring verification on TPU v5p2 matching the GPU VM golden reference (`nimble.py`) with zero numerical difference (`diff = 0.00e+00`).

## 3. Bugs Encountered & Solutions
1. **`to_maxtext.py` Merge Mode Invocation Trigger**:
   - `to_maxtext.py` determines merge mode via `is_merge_mode = bool(hf_lora_adapter_path and config.load_parameters_path)`.
   - Resolution: When running checkpoint conversion with `--lazy_load_tensors=true`, pass both `hf_lora_adapter_path=bespokelabs/Bespoke-Nimble-9B` and `load_parameters_path=gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-9b/unscanned/2026-09-23/0/items` (to signal merged mode). `_setup_merge_mode_getter` intercepts the tensor getter, adds the LoRA delta $W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (B @ A)$, and saves the fully merged checkpoint.
2. **`MaxEngine.prefill` Input Token Dimension**:
   - In `maxengine.py`, `_prefill_jit` performs `jnp.expand_dims(padded_tokens, 0)`.
   - Passing a 2D array `(1, max_prefill_predict_length)` resulted in a 3D tensor and triggered `assert decoder_input_tokens.ndim == 2`.
   - Resolution: `padded_tokens` must be passed as a 1D array of shape `(max_prefill_predict_length,)`.
3. **`ResultTokens` Token Extraction**:
   - `ResultTokens` returned by `engine.prefill` does not have a direct `.tokens` attribute; the token is retrieved via `result.get_result_at_slot(slot).tokens.item()`.

## 4. Model-Specific Insights & Gotchas
- **Zero Runtime Overhead**: Merging LoRA into base weights during `to_maxtext` conversion produces an ordinary unscanned Orbax checkpoint. Inference uses the standard optimized MaxText NNX engine without requiring any custom LoRA kernels during autoregressive decode.
- **Composite Linear Attention Weights**: In Qwen3.5, linear attention projects `qkv` and `z` into `in_proj_qkvz`, and `b` and `a` into `in_proj_ba`. `to_maxtext.py`'s merge getter correctly intercepts each individual sub-tensor (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`), applies their individual LoRA deltas, and passes the merged tensors to the concatenation hooks.

## 5. Verification Results vs Golden Reference

### Prompt Context
- **Context**: `"The store accepts returns within 30 days. This item was bought 12 days ago."`
- **Schema**: `eligible` (boolean: `"Is this item within the store return window?"`)
- **Candidates**:
  - `A` $\rightarrow$ `false` (Token ID: 32)
  - `B` $\rightarrow$ `true` (Token ID: 33)

### Comparison Table
| Metric | Golden Reference (GPU VM `nimble.py`) | MaxText TPU v5p2 (`verify_nimble_tpu.py`) | Discrepancy |
|---|---|---|---|
| **Prediction** | `{'eligible': True}` | `{'eligible': True}` | **Exact Match** |
| **Probability ('true' / 'B')** | `0.9998766054240137` | `0.9998766054240137` | **0.00e+00** |
| **Probability ('false' / 'A')** | `0.00012339457598623172` | `0.00012339457598623172` | **0.00e+00** |
| **Generated Token ID** | 33 (`'B'`) | 33 (`'B'`) | **Exact Match** |

## 6. Reproduction Commands

### Environment Setup
```bash
source /home/hengtaoguo_google_com/projects/venv1/bin/activate
cd /home/hengtaoguo_google_com/projects/maxtext
export HF_HOME=/dev/shm/hengtaoguo
export JAX_PLATFORMS=tpu
export PYTHONPATH=/home/hengtaoguo_google_com/projects/transformers/src:/home/hengtaoguo_google_com/projects/maxtext/src:/home/hengtaoguo_google_com/projects/maxtext
```

### 1. Checkpoint Conversion (LoRA Merged into Qwen3.5-9B Base Weights)
```bash
JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=qwen3.5-9b \
  base_output_directory=gs://hengtaoguo-maxtext-logs/checkpoints/bespoke-nimble-9b/unscanned/2026-09-23 \
  use_multimodal=false \
  scan_layers=false \
  weight_dtype=bfloat16 \
  hardware=cpu \
  skip_jax_distributed_system=True \
  checkpoint_storage_use_ocdbt=False \
  checkpoint_storage_use_zarr3=False \
  --lazy_load_tensors=true \
  hf_lora_adapter_path=bespokelabs/Bespoke-Nimble-9B \
  load_parameters_path=gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-9b/unscanned/2026-09-23/0/items \
  hf_access_token=<your_hf_token>
```
*Output Checkpoint*: `gs://hengtaoguo-maxtext-logs/checkpoints/bespoke-nimble-9b/unscanned/2026-09-23/0/items`

### 2. Autoregressive Decode on TPU v5p2
```bash
python3 -m maxtext.inference.decode \
  src/maxtext/configs/base.yml \
  run_name=decode_bespoke_nimble \
  model_name=qwen3.5-9b \
  tokenizer_path=Qwen/Qwen3.5-9B \
  load_parameters_path=gs://hengtaoguo-maxtext-logs/checkpoints/bespoke-nimble-9b/unscanned/2026-09-23/0/items \
  tokenizer_type=huggingface \
  prompt="Paris is the" \
  max_prefill_predict_length=16 \
  max_target_length=32 \
  per_device_batch_size=1 \
  ici_tensor_parallelism=4 \
  scan_layers=false \
  weight_dtype=bfloat16 \
  hf_access_token=<your_hf_token>
```

### 3. Golden Reference Candidate Verification Script
```bash
python3 /home/hengtaoguo_google_com/projects/verify_nimble_tpu.py
```

### 4. Run Autoregressive Decode on Exact Nimble Prompt via decode.py
```bash
python3 -m maxtext.inference.decode src/maxtext/configs/nimble.yml
```
Or via VS Code Launch Configuration:
- Configuration name: `decode_nimble` in `.vscode/launch.json`
- Generates: `Prediction: B` (matching choice `B` -> `true`).

### 5. MaxText Inference Module: NimbleModel Candidate Scoring (Exact Golden API & Format)

Located at `src/maxtext/inference/nimble.py` (with schema utilities at `src/maxtext/inference/parallel_schema.py`).

**Run via CLI**:
```bash
cd /home/hengtaoguo_google_com/projects/maxtext
python3 -m maxtext.inference.nimble
```

**Or import into Python scripts / pipelines**:
```python
from maxtext.inference.nimble import NimbleModel

model = NimbleModel()
result = model.score(
    context="The store accepts returns within 30 days. This item was bought 12 days ago.",
    schema={
        "eligible": {
            "type": "boolean",
            "description": "Is this item within the store return window?"
        }
    },
)

print(result["output"])
print(result["fields"]["eligible"]["probabilities"])
```

**Output**:
```python
{'eligible': True}
{'false': 0.00012339457598623172, 'true': 0.9998766054240137}
```

**VS Code Launch Configuration**:
- Configuration name: `nimble` in `.vscode/launch.json`
- Module: `maxtext.inference.nimble`
- CWD: `/home/hengtaoguo_google_com/projects/maxtext`



