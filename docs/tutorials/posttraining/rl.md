<!--
 Copyright 2024 Google LLC

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

# Reinforcement Learning on single-host TPUs

This tutorial demonstrates step-by-step instructions for setting up the
environment and then training the Llama3.1 8B-IT model on the GSM8K math
reasoning dataset using a single host TPU-VM such as `v6e-8/v5p-8`.

We utilize two RL algorithms, implemented via the Tunix library, to enhance the
model's reasoning capabilities:

- **Group Relative Policy Optimization (GRPO)**: GRPO is an RL algorithm
  designed to enhance the reasoning abilities of LLMs. It is a variant of
  Proximal Policy Optimization (PPO) that reduces memory usage by eliminating
  the need for a separate value function model. GRPO works by generating
  multiple responses for a given prompt, evaluating these responses using a
  reward model, and then calculating a relative advantage based on the group's
  performance to update the policy.

- **Group Sequence Policy Optimization (GSPO)**: GSPO is an RL algorithm that
  improves training efficiency and performance of LLMs by using sequence-level
  importance ratios and operations. GSPO defines the importance ratio based on
  sequence likelihood and performs sequence-level clipping, rewarding, and
  optimization.

For efficient model inference and response generation during this process, we
rely on the vLLM library.

Let's get started!

## TensorBoard metrics for standard GRPO and GSPO

With `enable_tensorboard=true`, MaxText writes held-out evaluation metrics
under `rl/eval/` to `tensorboard_dir`: `accuracy_percent`,
`partial_accuracy_percent`, `format_accuracy_percent`, `mean_reward`,
`num_prompts`, `num_samples`, `num_reward_samples`, `duration_seconds`, and
`success`. They are written before training, at `eval_interval`, and after
training, provided `num_test_batches > 0`. No additional model or dataset is
needed. A failed evaluation records `success=0` without manufacturing zero
accuracy. An empty evaluation records counts and `success=0`, with no accuracy.
Reward is omitted if no responses were successfully reward-scored.
The actual scored-response count is separate from the generated-sample count.

The evaluation x-axis is the completed outer rollout-policy step (including
the restored step when resuming). `rl/steps/actor_updates` records the actor's
optimizer-step counter at evaluation time on that same x-axis. This counter
can be ahead of the rollout policy: periodic evaluation runs from Tunix's
buffered train-metric hook. Existing Tunix `actor/train/*` tags retain their
optimizer-update x-axis; reward/completion tags retain their rollout-batch
x-axis. Do not equate these axes when using multiple updates per rollout.
Final and periodic evaluation can both produce a point at the final step.

`num_passes` records the configured number of samples per prompt. Accuracy
uses the existing `eval_mode` and sampling strategy; adding logging does not
change scoring or turn a non-greedy evaluation into a greedy one. In
particular, the legacy `pass` evaluator can overwrite an earlier correct but
malformed answer with a later wrong answer. Its result is therefore exported
as `accuracy_percent`, without claiming it is a conventional pass@k metric.

For MoE actors, add `rl.record_moe_metrics=true` to the training command to
enable `actor/train/moe/` diagnostics:

| Tag suffix                                  | Interpretation                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `router_entropy_mean_layer`                 | Mean per-token entropy of full-expert router scores, in nats, averaged across layers. |
| `router_entropy_min_layer`                  | Lowest layer's mean token entropy in each microbatch.                                 |
| `router_entropy_normalized_mean_layer`      | Router entropy divided by `log(num_experts)`.                                         |
| `expert_load_entropy_normalized_mean_layer` | Entropy of marginal expert assignments, divided by `log(num_experts)`.                |
| `expert_load_cv_mean_layer`                 | Standard deviation / mean expert assignment count.                                    |
| `max_to_mean_load_mean_layer`               | Maximum expert assignment count / mean count.                                         |
| `unused_expert_fraction_mean_layer`         | Fraction of routed experts receiving no valid assignments.                            |
| `valid_tokens_mean_layer`                   | Number of valid prompt and completion tokens observed per layer.                      |
| `observed_router_fraction`                  | Fraction of router modules supported by the observer.                                 |

Router entropy and expert-load entropy answer different questions: a
confident router can still distribute tokens evenly. Full-expert score
entropy is measured before top-k selection; sigmoid/positive router scores
are normalized before computing entropy, and DeepSeek expert-balancing bias
is excluded from the probability diagnostic. Load metrics use the selected
top-k assignments, including forced router replay when provided. Padding and
invalid replay ids are excluded; shared experts are not part of the routed
expert distribution.

These diagnostics use both valid prompt and completion tokens. Counts are
reduced across sharded token dimensions before computing load statistics.
Layer statistics are computed per training microbatch; Tunix averages those
statistics across microbatches in an optimizer update. They are not statistics
of the union of all update microbatches. Random, hash and fused routers are
unobserved and reflected in `observed_router_fraction`. Dense models emit no
MoE tags. This option currently supports the standard learner with unchunked
log-probability computation only. A single-expert router has zero entropy,
including the normalized diagnostic.

Observation adds a routing selection and compact per-token statistics, but
does not run the Transformer a second time, change router choices, add a
load-balancing penalty, or change the GRPO/GSPO objective. It is disabled by
default; measure its overhead on the accelerator and backend used for a run.

Existing `actor/train/entropy` measures output-token entropy for dense and
MoE models, not router entropy. Existing Tunix `perplexity` is the exponential
of the RL objective, not language-model perplexity. Prefer
`rewards/train/score/mean` for summed training reward;
`rewards/train/mean` averages reward components instead.

## Install MaxText and post-training dependencies

For instructions on installing MaxText with post-training dependencies on your VM, please refer to the [official documentation](../../install_maxtext.md) and use the `maxtext[tpu-post-train]` installation path to include all necessary post-training dependencies.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

## Setup environment variables

Login to Hugging Face. Provide your access token when prompted:
You can generate one at https://huggingface.co/settings/tokens.

```bash
hf auth login
```

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models. This value is case-sensitive.
export MODEL=<MODEL_NAME> # e.g. 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser) or via
# `gcloud storage buckets list --format="table(name, location)"`.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

# Number of accelerator chips per VM. For example, use 8 for a v5p-8 VM.
export CHIPS_PER_VM=<CHIPS_PER_VM>
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. It sets `MAXTEXT_CKPT_PATH`, which is passed to `load_parameters_path` in the training command below.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

Refer to [Hugging Face to MaxText](hf-to-maxtext) to convert a Hugging Face checkpoint to MaxText format. After conversion finishes, set `MAXTEXT_CKPT_PATH` to the converted MaxText checkpoint path.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

## Run GRPO

Run the following command for GRPO:

```
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  chips_per_vm=${CHIPS_PER_VM?}
```

The overview of what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of the model
   checkpoint you specified (e.g., `Llama3.1-8b-Instruct`).
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark
   after the post-training with GRPO.

By default, the above command will train the model using GRPOLearner from Tunix. To enable
asynchronous RL training with AgenticGRPOLearner, we can set `rl.use_agentic_rollout` to
true. An example command will be:

```
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  chips_per_vm=${CHIPS_PER_VM?} \
  rl.use_agentic_rollout=True
```

## Run GSPO

Run the following command for GSPO:

```
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  loss_algo=gspo-token \
  chips_per_vm=${CHIPS_PER_VM?}
```

The overview of what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of the model
   checkpoint you specified (e.g., `Llama3.1-8b-Instruct`).
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GSPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark
   after the post-training with GSPO.
