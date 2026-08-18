<!--
 # Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# Hy3 (Tencent Hunyuan V3)

Hy3 is an open-weights Mixture-of-Experts (MoE) model released by Tencent ([tencent/Hy3](https://huggingface.co/tencent/Hy3)).
* **Architecture**: 295B total parameters (~21B active parameters per token), 80 decoder layers with a dense 1st layer (`first_num_dense_layers: 1`) followed by 79 MoE layers.
* **Attention**: Grouped-Query Attention (GQA) with QK-Norm (RMSNorm on Query and Key head vectors) and RoPE.
* **MoE Routing**: Auxiliary-loss-free routing with Sigmoid activation and bias, 192 routed experts + 1 shared expert (selecting top-8 routed experts per token).
* **Supported Configs**: `hy3-tiny` (testing/smoke checks), `hy3-295b` (full-scale model).

### Note: MoE load balancing (previously broken, mostly fixed upstream)

Hy3's aux-loss-free routing (`routed_bias=true`) has two optional training-time
load-balancing mechanisms, controlled by `routed_bias_update_rate` (EMA-style
router-bias update) and `load_balance_loss_weight` (gradient-based auxiliary
loss). Both were broken in this framework (not specific to Hy3 -- the same
root cause reproduces on DeepSeek V3's `deepseek3-tiny` config) until two
recent upstream fixes landed:

- `1e6a5159f` ("[NNX] Preserve Intermediates in scanned layers for MoE load
  balance loss") fixed `scan_layers=true`: `nnx_decoders.py`'s scanned-layer
  application no longer discards `nnx.Intermediate` state (the sown
  `moe_bias_updates`/`moe_lb_loss` values) before `train.py` reads it.
- `263b8c18e` ("Fix shape mismatch for DeepSeek routed bias updates when MTP
  is enabled in NNX") fixed the previous total failure on `scan_layers=false`
  (crash / silently-missing intermediates): the bias-update path now finds
  the router bias generically via `_find_gate_bias` (searches the module
  graph by type, `GateLogit`) instead of assuming a hardcoded, scanned-only
  module path.

Verified with a CPU smoke test (`hy3-tiny`, `routed_bias_update_rate=0.05`,
`load_balance_loss_weight=0.01`):

- `load_balance_loss_weight` works correctly in both scan modes --
  `moe_lb_loss` is nonzero and changes every step regardless of
  `scan_layers`, since it's backpropagated through the normal loss/gradient
  path and never goes through the mechanism below.
- `routed_bias_update_rate` works correctly with `scan_layers=true`: the
  scanned decoder stores all MoE layers' biases as one stacked
  `(num_experts, num_moe_layers)` array, so there's no per-layer ambiguity.
  Confirmed by instrumenting each layer's own column of that array (not just
  the aggregate norm) over 4 steps (3 MoE layers): `[8.5e-05, 8.5e-05,
  8.5e-05] -> [0.1416, 0.1416, 0.1328] -> [0.2061, 0.2295, 0.1875] ->
  [0.2891, 0.1328, 0.2354] -> [0.2734, 0.2061, 0.1738]` -- three
  independently-evolving trajectories, confirming each layer gets its own
  correct update.
- **With `scan_layers=false`, `routed_bias_update_rate` is still broken**,
  just differently than before: each layer's own update signal is now
  collected correctly, but the code that applies it has two compounding
  bugs -- `getattr(decoder, "moe_layers", decoder)` falls back to the whole
  decoder (unscanned layers are attributes named `moe_layers_0`,
  `moe_layers_1`, ... individually, not a single stacked `moe_layers`), so
  `_find_gate_bias` only ever finds the *first* layer's bias; and the
  collection loop keeps overwriting a single `moe_bias_updates` variable, so
  only the *last* layer's own delta survives. Net effect, confirmed by
  instrumenting per-layer bias norms over 3 steps (3 MoE layers): layer 0's
  bias is updated every step, but with layer 2's delta (`8.5e-05 -> 0.1416
  -> 0.2832 -> 0.4258`); layers 1 and 2 never receive their own update at
  all (their bias only drifts by the normal optimizer gradient, ~1e-5/step,
  not the ~0.14/step jump the manual mechanism should produce). This is
  upstream `train.py` code shared with DeepSeek, not something specific to
  or fixable within this PR.

**Practical recommendation**: `load_balance_loss_weight` is safe to enable
in either scan mode. `routed_bias_update_rate` should only be enabled with
`scan_layers=true` until the `scan_layers=false` bug above is fixed
upstream -- enabling it with `scan_layers=false` today will silently update
the wrong layer's bias with the wrong values rather than error out.

`hy3-tiny.yml`/`hy3-295b.yml` leave both settings at their default (`0.0`);
turning either on by default isn't decided here.

---

## 1. Checkpoint Conversion

### Step 1: Download Model Weights from Hugging Face
Hy3's checkpoint is ~598GB, so make sure the target disk/host has enough free space
before downloading.

```bash
hf download tencent/Hy3 --local-dir /tmp/hy3-hf
```

Alternatively, `to_maxtext.py` defaults to `--lazy_load_tensors=True`, which fetches
tensors on demand directly from the `tencent/Hy3` HF Hub repo -- you can skip this
manual download step entirely and pass `model_name=hy3-295b` without
`--hf_model_path`/`--eager_load_method` in Step 2. This is the only practical option
for a partial/smoke conversion (e.g. a config truncated to a handful of layers via
`base_num_decoder_layers=N override_model_config=True`), since it only downloads the
specific shards needed instead of the full repo.

### Step 2: Convert to MaxText Orbax Format
Use MaxText's unified `to_maxtext` conversion tool to produce an Orbax checkpoint.

* **For Training / Fine-tuning (Scanned format, `scan_layers=true`)**:
```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=hy3-295b \
    scan_layers=true \
    attention=dot_product \
    base_output_directory=${BASE_OUTPUT_PATH} \
    hf_access_token=${HF_TOKEN} \
    hardware=cpu \
    skip_jax_distributed_system=True \
    --hf_model_path=/tmp/hy3-hf \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16
```

* **For Decoding / Inference (Unscanned format, `scan_layers=false`)**:
```bash
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=hy3-295b \
    scan_layers=false \
    attention=dot_product \
    base_output_directory=${BASE_OUTPUT_PATH} \
    hf_access_token=${HF_TOKEN} \
    hardware=cpu \
    skip_jax_distributed_system=True \
    --hf_model_path=/tmp/hy3-hf \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16
```

---

## 2. Pre-training

### Smoke Test (Local / Single TPU / CPU with `hy3-tiny`)
```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=hy3-tiny \
    steps=10 \
    per_device_batch_size=1 \
    dataset_type=synthetic \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=hy3_tiny_smoke_test
```

### Full-Scale Pre-training (`hy3-295b` on Multi-Slice TPU v5p)
Example training run on TPU v5p-256 / v5p-512:
```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=hy3-295b \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=hy3_295b_pretraining \
    per_device_batch_size=1 \
    max_target_length=4096 \
    ici_fsdp_parallelism=64 \
    ici_expert_parallelism=4 \
    megablox=true \
    sparse_matmul=true \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    dataset_type=synthetic \
    steps=100
```

---

## 3. Fine-tuning

After converting the checkpoint to scanned Orbax format, you can fine-tune with your own dataset:
```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=hy3-295b \
    load_parameters_path=${CONVERTED_ORBAX_PATH} \
    scan_layers=true \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=hy3_295b_sft \
    per_device_batch_size=1 \
    max_target_length=4096 \
    dataset_type=huggingface \
    hf_path=${HF_DATASET_PATH} \
    learning_rate=2e-5 \
    steps=1000
```

---

## 4. Inference / Text Generation (Decode)

To perform text generation using the unscanned checkpoint:
```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    model_name=hy3-295b \
    load_parameters_path=${CONVERTED_ORBAX_UNSCANNED_PATH} \
    scan_layers=false \
    max_prefill_predict_length=64 \
    max_target_length=256 \
    prompt="Hello, who are you?"
```

---

## 5. Verification: Forward Pass Logit Check

Verify numerical equivalence between MaxText and Hugging Face reference:
```bash
python3 tests/utils/forward_pass_logit_checker.py \
    src/maxtext/configs/base.yml \
    --run_hf_model=True \
    --hf_model_path=/tmp/hy3-hf \
    model_name=hy3-295b \
    scan_layers=false \
    weight_dtype=float32 \
    dtype=float32 \
    activations_in_float32=true \
    matmul_precision=float32 \
    float32_logits=true \
    float32_qk_product=true
```

**Note on full-scale (80-layer) runs:** `--run_hf_model=True` loads the real
Hugging Face PyTorch reference model on the host CPU (not on TPU chips), so it
needs enough host RAM to hold the full model -- roughly 590GB for Hy3's 295B
parameters in bf16 (more in float32, as used by the command above). A standard
single TPU-VM host will not have enough RAM for this; you'd need either a
large-memory VM or a sharded/distributed PyTorch loading setup. For a
resource-friendly sanity check, truncate both the MaxText config and the HF
reference config to a handful of layers (e.g. `base_num_decoder_layers=2
override_model_config=True` on the MaxText side, and
`AutoConfig.from_pretrained(..., num_hidden_layers=2)` plus a golden-logits jsonl
generated from that truncated HF model on the reference side, since
`--run_hf_model=True` does not support truncating the live-loaded reference model)
-- since Hy3 has no per-layer-varying architecture, a truncated-layer comparison
still exercises every distinct code path (attention, dense MLP, MoE routing,
shared expert).
