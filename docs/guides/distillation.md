<!--
 Copyright 2023-2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Distillation

This guide covers how MaxText's online distillation trainer works, the loss anatomy, the configuration surface, and how to tune the loss-weight schedules (α, β, temperature) for different scenarios.

For step-by-step launch recipes (single-host and multi-host), see the [Knowledge Distillation tutorial](../tutorials/posttraining/knowledge_distillation.md).

## Overview

MaxText supports two flavors of knowledge distillation:

1. **Offline distillation** — the teacher generates a dataset (or top-k logits) once; the student is trained on the cached output. Cheapest when teacher inference is expensive and you plan to run multiple student experiments.
2. **Online distillation** — teacher and student share the same training loop and the teacher runs forward each step. Required when you want **feature-level alignment** (intermediate activations) and useful for same-family compression and pruning recovery.

This guide focuses on the **online trainer**, [`maxtext.trainers.post_train.distillation.train_distill`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/train_distill.py), which is built on [Tunix](https://github.com/google/tunix). Common use cases:

- **Same-size pruning recovery** — recover quality after structural pruning by aligning logits (and optionally activations) to the unpruned teacher.
- **Compression** — distill a larger teacher into a smaller student of the same family (e.g. Llama-3.1-70B → Llama-3.1-8B).
- **Self-distillation** — improve a model by distilling it from itself with stronger regularization or a different data mix.

### Online vs. offline at a glance

|                        | Online                                              | Offline                                                      |
| ---------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Teacher inference cost | Per training step                                   | One-time data generation                                     |
| Storage cost           | None beyond checkpoints                             | Significant (full dataset of teacher outputs)                |
| Hardware required      | Both teacher + student fit in mesh                  | Student only during training                                 |
| Supports feature loss  | Yes (`distill_beta > 0`)                            | No (only logit-level)                                        |
| Best for               | Same-family pruning recovery, small/medium teachers | Very large teachers, repeat student experiments on same data |

A hybrid pattern — **cache top-k teacher logits offline, then run the trainer in offline mode** — is also supported via `save_top_k_teacher_logits.py` and the `offline_data_dir` flag. See the tutorial for the recipe.

## Architecture

The trainer initializes **two MaxText models** with isolated configurations:

- **Student** — trainable; configured from the YAML plus `student_overrides`.
- **Teacher** — frozen (`stop_gradient`); configured from the YAML plus `teacher_overrides`.

This separation lets you use the same base config for both while still varying e.g. `model_name`, `num_decoder_layers`, or `load_parameters_path` per side. CLI overrides only apply to the **student** by default — the teacher is initialized from the YAML + `teacher_overrides` only, so flags like `num_query_heads=16` passed on the command line will not silently change the teacher.

### Vocabulary requirement

Student and teacher must share the same vocabulary. The trainer asserts `student_config.vocab_size == teacher_config.vocab_size` at startup.

### Required architectural flags for feature loss

If `distill_beta > 0`, the model `sow`s the attention `out_projection` activations at every layer so the loss can read them. This requires:

- `scan_layers: True` — activations are stacked along the leading scan axis; the loss does `jnp.take(features, layer_indices, axis=0)` over that axis.
- `enable_nnx: True` — `sow(nnx.Intermediate, ...)` is an NNX-specific call.

The trainer validates both at config initialization. Logit-only runs (`distill_beta = 0`) have no such constraint.

## Loss anatomy

The total per-step loss is:

```
L_total = α · KL(teacher_T || student_T) · T²   ←  soft loss
        + (1 − α) · CE(student, labels)         ←  hard loss
        + β · feature_loss(student_acts, teacher_acts[layer_indices])
```

Where `T` is the temperature, KL is over softmax-with-temperature distributions, and `feature_loss` is mean cosine distance (default) or L2.

The Hinton T² scaling is applied automatically inside `compute_loss`, so the soft-loss magnitude stays comparable as you change T.

Per-token validity is derived from the one-hot labels — padded positions (fully-zero rows) are excluded from the loss. All token-weighted metrics are emitted as `(sum, count)` pairs and aggregated as `sum(sums) / sum(counts)`, so the values are unbiased across multi-host averaging and across logging windows with varying valid-token counts.

## Configuration surface

The starter config is [`src/maxtext/configs/post_train/distillation.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/post_train/distillation.yml). Its key sections:

```yaml
base_config: "base.yml"

# Student and teacher are configured separately; CLI args only flow into the student.
student_overrides:
  model_name: "llama3.1-8b"

teacher_overrides:
  model_name: "llama3.1-8b"
  load_parameters_path: "/path/to/teacher/checkpoint/0/items"   # required for online runs

# --- Logit distillation ---
distill_alpha: 0.5             # weight on KL(teacher||student)
distill_temperature: 1.0       # softmax temperature applied before KL

# --- Feature distillation (optional; 0.0 disables) ---
distill_beta: 0.0
distill_feature_loss_type: "cosine"   # or "l2"
distill_layer_indices: None           # which scanned layers to align

# --- Schedules — when *_end is None, the value stays fixed ---
distill_alpha_end: None
distill_alpha_schedule: "constant"    # constant | linear | cosine
distill_temperature_end: None
distill_temperature_schedule: "constant"
distill_beta_end: None
distill_beta_schedule: "constant"
```

### Schedule semantics

`progress = clip(step / max_steps, 0, 1)`. Past `max_steps`, the value freezes at `end_value`.

- `constant` — fixed at `start_value`; `end_value` ignored.
- `linear` — `start + (end − start) · progress`.
- `cosine` — `end + (start − end) · 0.5 · (1 + cos(π · progress))`. Holds near `start` longer than linear before transitioning.

## α (alpha) schedule guide

α weights the **soft KL loss** against the **hard CE loss**:

- `α = 1.0` → pure teacher mimicry (KL only)
- `α = 0.0` → pure SFT (CE only)
- α weights how much you trust the **teacher's distribution** vs the **one-hot label**

### Why decay high → low

| Phase             | What's happening                                                                    | Right α            |
| ----------------- | ----------------------------------------------------------------------------------- | ------------------ |
| Recovery (early)  | Student damaged by pruning; teacher's full softmax is dense, info-rich, low-noise   | High (0.8–1.0)     |
| Refinement (late) | Student close to teacher; KL diminishing returns; teacher's errors start to bake in | Moderate (0.3–0.5) |

### Recommended α schedules

| Scenario                                                                | start | end | schedule |
| ----------------------------------------------------------------------- | ----- | --- | -------- |
| Large teacher → small student (e.g. 70B → 8B)                           | 1.0   | —   | constant |
| Same-size pruning recovery (default recommendation)                     | 0.9   | 0.5 | cosine   |
| Same-size, clean labels, want student > teacher on label-grounded tasks | 0.9   | 0.3 | cosine   |
| Reasoning / code (label is gospel)                                      | 0.8   | 0.2 | cosine   |
| Offline top-k logits (narrow teacher support)                           | 0.7   | 0.3 | cosine   |
| Conservative baseline (current default)                                 | 0.5   | —   | constant |

Prefer `cosine` over `linear` when decaying — cosine holds near `start_value` longer before transitioning, which better matches recovery dynamics.

## β (beta) schedule guide

β scales an **additive** feature-loss term — unlike α, it doesn't trade off against another loss. Increasing β just adds more pressure to align the student's attention out-projection activations to the teacher's at the chosen layers.

Because it's additive, β's *absolute* magnitude matters relative to the logit losses:

- The cosine-distance feature loss is bounded in `[0, 2]` per element → `β` of order **0.1–2.0** is typical.
- L2 is unbounded → use `β` of order **0.01–0.1**.

**Decay β high → low**: high β early forces the student's internals to match the teacher (strongest recovery signal); decay it as the student converges so the rigid same-shape activation match doesn't become a ceiling.

### Recommended β schedules

| Scenario                                                    | feature_loss_type | start | end   | schedule |
| ----------------------------------------------------------- | ----------------- | ----- | ----- | -------- |
| Off (logit-only distillation)                               | —                 | 0.0   | —     | constant |
| Same-size pruning recovery (default)                        | cosine            | 1.0   | 0.1   | cosine   |
| Aggressive recovery (heavy pruning)                         | cosine            | 2.0   | 0.5   | cosine   |
| L2 variant                                                  | l2                | 0.05  | 0.005 | cosine   |
| Constant feature pressure (architecturally similar student) | cosine            | 0.5   | —     | constant |

> **Note:** `distill_beta = 0.0` disables feature *extraction* entirely (the `sow` is skipped), so you cannot start at 0 and ramp up. To "ramp on" feature loss, start at a tiny positive value (e.g. `1e-6`) and set `distill_beta_end` to your target.

### Layer indices for feature loss

`distill_layer_indices` selects which scanned-layer slices contribute to `feature_loss`. The XPK launcher's default is `[0,1,2,...,7]` — the first 8 layers, irrespective of model depth. Better defaults usually exist:

| Goal                                        | Llama-8B (32 layers)                        | Llama-70B (80 layers)      |
| ------------------------------------------- | ------------------------------------------- | -------------------------- |
| Anchor low-level (current launcher default) | `[0,1,2,3,4,5,6,7]`                         | `[0,1,2,3,4,5,6,7]`        |
| Cover full depth (recommended)              | `[3,7,11,15,19,23,27,31]`                   | `[9,19,29,39,49,59,69,79]` |
| Top-heavy (semantic layers matter most)     | `[16,20,24,28,30,31]`                       | `[60,65,70,75,78,79]`      |
| Bracket pruned region                       | layers immediately before/after pruned ones | same                       |

If you have pruned specific layers, the most useful targets are usually the layers **straddling the pruned region** — those representations are the most disturbed.

## Temperature schedule

Higher T softens the distributions and transfers more "dark knowledge" (relative ordering of non-top tokens).

| T                     | Effect                                                     |
| --------------------- | ---------------------------------------------------------- |
| 1                     | Raw softmax; fastest convergence on the dominant token     |
| 2 (recommended start) | Meaningful contribution from non-top tokens                |
| 4+                    | Very flat; soft-loss gradient shrinks even with T² scaling |

Common pattern: anneal T from 2.0 → 1.0 alongside α decay.

## Starter configurations

### Same-size pruning recovery (default recommendation)

```yaml
# logits
distill_alpha: 0.9
distill_alpha_end: 0.5
distill_alpha_schedule: cosine
distill_temperature: 2.0
distill_temperature_end: 1.0
distill_temperature_schedule: cosine

# features
distill_beta: 1.0
distill_beta_end: 0.1
distill_beta_schedule: cosine
distill_feature_loss_type: cosine
distill_layer_indices: [3, 7, 11, 15, 19, 23, 27, 31]   # for 32-layer student

scan_layers: True
enable_nnx: True
```

### Logit-only baseline (cheapest; no feature extraction overhead)

```yaml
distill_alpha: 0.9
distill_alpha_end: 0.5
distill_alpha_schedule: cosine
distill_temperature: 2.0
distill_beta: 0.0            # disables sow; no extra memory or compute
```

For other shapes (large teacher → small student, aggressive recovery, etc.), adjust `distill_alpha`/`distill_beta` per the [α](#recommended-%CE%B1-schedules) and [β](#recommended-%CE%B2-schedules) schedule tables above.

## Monitoring

The trainer logs the following to TensorBoard (configured by `tensorboard_dir`, defaulting to a path under `base_output_directory`):

| Metric                                                         | What it tells you                                                                                                                |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `distill/soft_loss`                                            | KL on temperature-softened distributions, scaled by T². The soft-loss component of the gradient.                                 |
| `distill/hard_loss`                                            | Student CE on labels. Should track the teacher's after recovery.                                                                 |
| `distill/teacher_loss`                                         | Teacher CE on labels — sanity check; should be ~constant. Jumping means the batch composition changed or the teacher mis-loaded. |
| `distill/student_perplexity`, `distill/teacher_perplexity`     | Per-step next-token perplexity. The convergence gap is the student↔teacher quality gap.                                          |
| `distill/kl_div_at_T`                                          | KL at the current (scheduled) temperature, without the T² scaling.                                                               |
| `distill/kl_div_T1`                                            | KL at T=1. Comparable across runs / different temperature schedules. **Best metric for cross-run quality comparison.**           |
| `distill/out_proj_feature_loss`                                | The feature-loss term (already β-scaled). Should drop early then plateau.                                                        |
| `distill/total_loss`                                           | The full optimization target.                                                                                                    |
| `distill/alpha`, `distill/beta_feature`, `distill/temperature` | Confirms the schedulers are firing as intended.                                                                                  |

> **Note:** The headline `_train_perplexity` Tunix prints is `exp(total_loss)` which for distillation is `exp(α·soft + (1-α)·hard + β·feature)` — **not** next-token perplexity. Use `distill/student_perplexity` or compute `exp(distill/hard_loss)` for the comparable quality number.

## Troubleshooting

| Symptom                                                                             | Likely cause                                                                                           | Fix                                                                                                                           |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: a value of self.distill_beta > 0.0 requires self.scan_layers = True`   | Feature loss enabled without scanned layers.                                                           | Add `scan_layers=True enable_nnx=True` to your CLI / yml.                                                                     |
| `Vocab size mismatch! Student: X, Teacher: Y`                                       | Different tokenizers.                                                                                  | Use teacher and student with the same vocab; the trainer cannot match logits across vocabularies.                             |
| `Teacher model path is missing`                                                     | `teacher_overrides.load_parameters_path` not set in non-offline mode.                                  | Set it in `teacher_overrides` in the yml or pass via CLI.                                                                     |
| `Features extracted from student or teacher model are None, but distill_beta > 0.0` | Model architecture doesn't sow `out_projection_activations` (e.g. uses an unsupported attention path). | Verify the attention layer in use sets `self.sow(nnx.Intermediate, "out_projection_activations", out)` (see `attentions.py`). |
| `distill_beta=0.0 but distill_beta_end=...`                                         | Trying to ramp β up from zero, but `0.0` disables the `sow` so there's nothing to ramp.                | Start at a small positive value (e.g. `distill_beta=1e-6`) and set `distill_beta_end` to the target.                          |
| `hard_loss` keeps rising while `soft_loss` drops                                    | α is locking the student into teacher behavior that's bad against labels.                              | Decay α faster (lower `distill_alpha_end`), or decrease the alpha curve mid-training.                                         |
| `out_proj_feature_loss` plateaus high and won't drop                                | Wrong layer indices for the pruning pattern, or β too high (numerically dominating gradients).         | Re-examine `distill_layer_indices`; lower starting β or its end value.                                                        |
| `kl_div_T1` plateaus quickly then nothing improves                                  | Student capacity-bound on those layers; or teacher distribution too narrow.                            | Raise β to push deeper alignment; revisit the prune; or raise temperature.                                                    |
| Teacher OOMs at startup                                                             | Teacher is too large for the mesh + student.                                                           | Use the offline top-k variant; or reduce `per_device_batch_size`; or move to a larger slice.                                  |

## Ablation priority

When tuning a new run, ablate in this order — each is a config-only change with no code edits:

1. **`distill_alpha_end`** — try {0.3, 0.5, 0.7} with `start=0.9`, `schedule=cosine`. Highest-leverage knob.
2. **`distill_layer_indices`** *(only if `distill_beta > 0`)* — evenly-spaced vs first-8 vs straddling pruned layers. Often as impactful as β value.
3. **`distill_beta_end`** *(only if `distill_beta > 0`)* — {0.01, 0.1, 0.5} from `start=1.0`. Low end = "let internals drift", high end = "enforce alignment".
4. **`distill_temperature`** — {1.0, 2.0, 4.0} starting values. T=2 is usually safe.
5. **Schedule shape** — `cosine` vs `linear` for α. Cosine usually wins.
6. **`distill_feature_loss_type`** — `cosine` vs `l2`. Cosine is more forgiving; L2 punishes magnitude drift too.

## Related

- [Knowledge Distillation tutorial](../tutorials/posttraining/knowledge_distillation.md) — step-by-step launch recipes.
- [Optimization guide](optimization.md) — sharding strategies and performance tuning that apply to the distillation trainer too.
