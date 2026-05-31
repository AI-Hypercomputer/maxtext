# Qwen3-30B-A3B Learn-to-Init Distillation — Setup

End-to-end steps to run Learn-to-Init (LTI) soft distillation from a
Qwen3-30B-A3B-base teacher to a custom student (half query/KV heads, doubled
head_dim) in MaxText.

- **Teacher**: `qwen3-30b-a3b-base` (converted MaxText checkpoint)
- **Student**: custom variant — `learn_to_init_mode: True`
- **Recipe**: `src/maxtext/configs/post_train/distillation_qwen3_30b_lti.yml`
- **Hardware tested**: TPU v7x-8 (8 chips, 96 GB HBM/device)

---

## 1. Environment

Editable MaxText install plus TPU extras:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[tpu]'
.venv/bin/install_tpu_pre_train_extra_deps
```

Verify TPU visibility:

```bash
.venv/bin/python -c "import jax; print(len(jax.devices()), jax.devices()[0])"
```

### Add tunix (required for the distillation trainer)

The `[tpu]` extra does not include tunix. Distillation lives under
`maxtext.trainers.post_train.distillation`, which imports `tunix`. Install
the pinned tunix sha matching MaxText's canonical XPK image
(`run_distill_xpk.sh:prep_image`), then re-pin libtpu/jax to keep
image/libtpu/jax compatibility:

```bash
.venv/bin/python -m pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/google/tunix@<pinned-sha>"

.venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
  jax==0.10.0 jaxlib==0.10.0 libtpu==0.0.39
```

> The `tpu-post-train` extra pulls in vLLM and tpu-inference (large) and
> downgrades `flax`/`optax`. The two commands above install only what the
> distillation trainer needs.

---

## 2. Convert the teacher checkpoint

Convert the HF Qwen3-30B-A3B-base weights to MaxText format using the
unified `to_maxtext` script:

```bash
python -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml \
  model_name=qwen3-30b-a3b-base \
  load_parameters_path=<hf-checkpoint-path> \
  base_output_directory=<gcs-or-local-out-path> \
  hardware=cpu skip_jax_distributed_system=True scan_layers=True
```

The custom student is materialized at training time from this teacher
checkpoint via the recipe's `student_overrides` and the copy_map (Section 5);
no separate student conversion is needed.

---

## 3. The distillation config

`src/maxtext/configs/post_train/distillation_qwen3_30b_lti.yml` enables LTI
distillation. Key fields:

### Custom student shape

```yaml
student_overrides:
  model_name: "qwen3-30b-a3b-base"
  override_model_config: True
  base_num_query_heads: 16     # teacher: 32
  head_dim: 256                # teacher: 128
  base_num_kv_heads: 2         # teacher: 4
```

`rope_max_timescale` is inherited from the base model config (1e7), applied
at the new head_dim. The A,B bridges learn to adapt to whatever RoPE
frequencies are present.

### LTI mode

```yaml
learn_to_init_mode: True
attn_module_name: "self_attention"
lti_use_general_linear_map: False   # bilinear bridge; cheaper HBM
```

### YAML top-level requirement

Batch-shape fields (`per_device_batch_size`, `gradient_accumulation_steps`,
`max_target_length`) must be set at the YAML top level — the trainer
rebuilds the teacher config from the YAML only, and a shape mismatch trips
a validator at startup.

---

## 4. Smoke test

End-to-end pipeline check (LTI swap + forward + loss + ckpt) at small
batch/seq:

```bash
.venv/bin/python -m maxtext.trainers.post_train.distillation.train_distill \
  src/maxtext/configs/post_train/distillation_qwen3_30b_lti.yml \
  run_name=smoke-lti-$(date +%Y%m%d-%H%M%S) \
  base_output_directory=<output_path> \
  max_target_length=2048 \
  steps=20 checkpoint_period=10
```

With an empty `distill_weights_copy_map`, expect:

```
total_loss ~ 10      soft_loss = kl_div_T1 ~ 10      hard_loss ~ 12
student_perplexity ~ 2.5e+05    teacher_perplexity ~ single-digit
moe_lb_loss ~ 0.02
```

- `student_perplexity` near vocab size = near-random — expected because
  non-attention weights are randomly initialized.
- `total_loss ≈ soft_loss` because `distill_alpha=1.0` (pure KD).

Compile is ~10–15 min cold, ~1 min if `~/workspace/jax_cache` is warm.

---

## 5. Derive `distill_weights_copy_map`

`distill_weights_copy_map` tells `lti_utils.prepare_student_weights` which
teacher tensors to copy into the student at init. Without it, only LTI's
internal bridges are randomly initialized — every non-attention weight is
random too, and loss starts far above the floor.

A helper script
`src/maxtext/trainers/post_train/distillation/tools/derive_lti_copy_map.py`
uses `nnx.eval_shape` (no weights materialized) to walk both abstract
graphs and emit a copy_map for every path whose shape exactly matches:

```bash
.venv/bin/python -m maxtext.trainers.post_train.distillation.tools.derive_lti_copy_map \
  src/maxtext/configs/post_train/distillation_qwen3_30b_lti.yml \
  > /tmp/copy_map.yml
```

Inspected skips are expected: attention q/k/v/out projections (wrapped in
`LearnToInitDense`) and q_norm/k_norm (shape depends on head_dim which
differs). Paste the `distill_weights_copy_map: ...` block into the YAML.

Critically, the copy map must also copy teacher attention kernels into
the student's frozen `C` buffer:

```yaml
distill_weights_copy_map:
  "decoder/layers/self_attention/query/kernel": "decoder/layers/self_attention/query/C"
  "decoder/layers/self_attention/key/kernel":   "decoder/layers/self_attention/key/C"
  "decoder/layers/self_attention/value/kernel": "decoder/layers/self_attention/value/C"
  "decoder/layers/self_attention/out/kernel":   "decoder/layers/self_attention/out/C"
```

Without this, `C` stays at `jnp.empty()` (≈zero) and the LTI bridges
compute `A · 0 · B = 0`, so attention output is zero.

---

## 6. Run

After the copy map is in the YAML:

```bash
.venv/bin/python -m maxtext.trainers.post_train.distillation.train_distill \
  src/maxtext/configs/post_train/distillation_qwen3_30b_lti.yml \
  run_name=qwen3-30b-lti-$(date +%Y%m%d-%H%M%S) \
  base_output_directory=<output_path>
```

### Expected timings (TPU v7x-8)

- Teacher checkpoint load: ~4 min
- Student init + LTI weight injection: ~5 s (after teacher is loaded)
- XLA compile: ~1 min warm cache; ~10–15 min cold
- Step time (per_device=1, grad_accum=1, seq=4096): **~1.8 s/step**
- 64 000 steps: roughly **~32 h wall-clock**
- Checkpoint save to GCS: ~5 min per save (async — overlaps with training)

### Memory expectations (TPU v7x-8, 96 GB HBM/device)

Per-device rough budget (FSDP shards across 8 devices):

- Teacher params (bf16, frozen): ~7.5 GB
- Student params (bf16): ~7.5 GB
- Adam optimizer state (fp32 m + fp32 nu, student only): ~30 GB
- **Static state per device: ~45 GB**
- Activations (seq 4096, batch 1, fp32 logits): ~20 GB peak
- **Total per device: ~65 GB / 96 GB cap → ~30 GB headroom**

---

## 7. Outputs

For each run, the trainer writes under `<base_output_directory>/<run_name>/`:

- `distillation.yml` — verbatim copy of the source YAML
- `command.sh` — pasteable command with CLI overrides
- `checkpoints/<step>/` — Orbax model_params + iter
- `tensorboard/` — TensorBoard event files

Resume a crashed run: re-launch with the same `run_name` and
`base_output_directory`; the trainer auto-restores from the latest
checkpoint.

