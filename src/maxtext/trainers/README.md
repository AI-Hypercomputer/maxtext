# MaxText trainers

This package holds MaxText's training entry points, grouped by stage
(`pre_train`, `post_train`, etc.). Each trainer is a runnable module launched
with `python -m maxtext.trainers.<...>`. Most trainers take a config YAML plus
`key=value` overrides.

This README is a map of the package. For concepts and step-by-step guides, see
the [documentation website](https://maxtext.readthedocs.io/en/latest/).

## Layout

| Package | Purpose |
|---|---|
| `pre_train` | Core pretraining/finetuning loop (`train.py`) and ahead-of-time compilation (`train_compile.py`) |
| `diloco` | DiLoCo low-communication training strategy: a library layered onto the pretraining loop |
| `post_train` | Post-training regimes: SFT, DPO, RL (GRPO/GSPO), and knowledge distillation |
| `tokenizer` | Train a SentencePiece tokenizer from a Grain dataset |

## Entry points

Most runnable modules (except for `tokenizer`; more details below) have a
default config from [`configs/pyconfig.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/pyconfig.py) that can be
overridden on the command line. Modules with no entry in that mapping fall back
to `base.yml` with a warning, so pass a config path explicitly when `base.yml`
is not the right config.

For example, to run the pretraining loop with a custom run name:

```bash
python -m maxtext.trainers.pre_train.train run_name=<run_name>
```

Pass a YAML as the first positional argument to override the default:

```bash
python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml run_name=<run_name>
```

### Pretraining — `pre_train`

Train a model from scratch or continue training an existing checkpoint on raw
text with a next-token prediction loss. `train.py` is MaxText's native training
loop and supports both Flax Linen (`nn.Module`) and Flax NNX models;
`train_compile.py` does not train, but compiles that loop's train step for a
target topology.

- `pre_train.train`: uses `base.yml` as default config file. See
  [pretraining.md](https://maxtext.readthedocs.io/en/latest/tutorials/pretraining.html) for a step-by-step
  guide.
- `pre_train.train_compile`: uses `base.yml` as default config file. Performs
  ahead-of-time (XAOT) compilation. See
  [Features and diagnostics](https://maxtext.readthedocs.io/en/latest/guides/monitoring_and_debugging/features_and_diagnostics.html)
  for more details.

### DiLoCo — `diloco`

DiLoCo is a low-communication training strategy layered onto the pretraining
loop. It is gated by `enable_diloco` and the `diloco_*` flags in `base.yml`, and
is consumed via [`utils/train_utils.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/train_utils.py). DiLoCo
requires NNX models and has no entry point of its own; set `enable_diloco=True`
on a pretraining run to use it. For example,

```bash
python -m maxtext.trainers.pre_train.train \
  enable_diloco=true \
  dcn_diloco_parallelism=2 \
  run_name=<run_name>
```

### Post-training — `post_train`

Adapt an already-pretrained checkpoint: follow instructions (SFT), align to
preference pairs (DPO), optimize against a reward signal (RL), or train a
student model against a teacher's outputs (distillation). Most of these wrap
MaxText models in [Tunix](https://github.com/google/tunix) trainers rather than
reusing the native loop: `MaxTextPeftTrainer` (SFT) and
`MaxTextDistillationTrainer` (distillation) subclass Tunix's `PeftTrainer`, and
DPO uses Tunix's `DPOTrainer` directly.

| Regime | Module | Default config | Docs |
|---|---|---|---|
| SFT | `post_train.sft.train_sft` | `post_train/sft.yml` | [sft.md](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft.html) |
| SFT (native loop) | `post_train.sft.train_sft_native` | `post_train/sft.yml` | [sft.md](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft.html) |
| DPO | `post_train.dpo.train_dpo` | `post_train/dpo.yml` | [dpo.md](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/dpo.html) |
| RL (GRPO, GSPO) | `post_train.rl.train_rl` | `post_train/rl.yml` | [rl.md](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html) |
| Distillation | `post_train.distillation.train_distill` | `post_train/distillation.yml` | [distillation README](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/README.md), [knowledge_distillation.md](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/knowledge_distillation.html) |

**Native-loop SFT** — `train_sft_native` reuses `pre_train/train.py`, appending
`use_sft=True` and `use_tunix_gradient_accumulation=False` to the config
overrides.

**RL variants** — a single `train_rl` covers GRPO (the default) and GSPO
(`loss_algo=gspo-token`). A separate legacy implementation lives outside this
package at [`experimental/rl`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/experimental/rl/README.md); it takes two
config files (train + inference) and has no default-config mapping. New work
should use `post_train.rl.train_rl`.

**Hooks** — `post_train/hooks.py` defines `BaseTrainingHooks` and
`BaseDataHooks`; SFT and DPO subclass both (`sft/hooks.py`, `dpo/hooks.py`). RL
is the exception: `rl/hooks.py` extends Tunix's `TrainingHooks` directly to fire
`evaluate(...)` every `eval_interval` outer steps, and defines no data hooks.

**Distillation** — `train_distill` loads dual student/teacher models, with an
optional Learn-to-Init phase gated by `learn_to_init_mode` (declared in
[`configs/types.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/types.py) rather than in any shipped YAML). The
offline top-k logit pipeline below is runnable but not a trainer, and has no
entry in the config mapping, so pass a config explicitly:

| Module | Docs |
|---|---|
| `post_train.distillation.save_top_k_teacher_logits` | [distillation README](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/README.md) |
| `post_train.distillation.verify_saved_logits` | [distillation README](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/README.md) |

### Tokenizer — `tokenizer`

Fit a SentencePiece vocabulary over a Grain corpus (`parquet`, `arrayrecord`,
or `tfrecord`) and write the tokenizer model to the assets directory, ready to
feed a pretraining run. Configured entirely with absl flags, rather than a
config YAML (`--grain_train_files` is required).

Example usage:

```bash
python -m maxtext.trainers.tokenizer.train_tokenizer \
  --grain_train_files=gs://my-bucket/data/*.parquet \
  --grain_file_type=parquet
```

## Related docs

- [Model configs](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/README.md) — tuned per-hardware model configs and XLA flags
- [Pretraining](https://maxtext.readthedocs.io/en/latest/tutorials/pretraining.html)
- [Post-training index](https://maxtext.readthedocs.io/en/latest/tutorials/post_training_index.html)
- [Distillation guide](https://maxtext.readthedocs.io/en/latest/guides/distillation.html)
