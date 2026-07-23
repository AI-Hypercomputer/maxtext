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

# Reinforcement Learning with gemma4-26b on Multi-Host TPUs

This tutorial provides step-by-step instructions for setting up the environment
and training the gemma4-26b (26B-A4B MoE) model with GRPO on the [OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) on a Cloud TPU v7x (Ironwood) GKE cluster using a `tpu7x-4x4x4` (64-chip) slice.

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- A Hugging Face account with an access token for downloading models (the `google/gemma-4-26B-A4B` and `google/gemma-4-26B-A4B-it` repositories are gated; request access before proceeding).
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- Prerequisites for XPK installed (follow [official documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md#1-prerequisites)).
- A Pathways-ready GKE cluster (see [create GKE cluster](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)).
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Setup Environment Variables

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# Your GCP project ID.
# If you've already set it in your local config, you can retrieve it via:
# gcloud config get-value project
export PROJECT_ID=<PROJECT_ID>

# The name of your GKE cluster.
export CLUSTER_NAME=<CLUSTER_NAME>

# The GCP location of your GKE cluster.
export ZONE=<ZONE> # e.g., 'us-central1' or 'us-central1-a'

# Use a GCS bucket you own to store logs and checkpoints.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs
```

## Authenticate with Hugging Face

To download the `gemma4-26b` model checkpoint from Hugging Face, you need to authenticate using your Hugging Face account credentials. Run the following command and follow the prompts to log in:

```bash
hf auth login
```

## Get Your MaxText Compatible Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

Refer to [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a Hugging Face checkpoint to MaxText format. After conversion finishes, set `MAXTEXT_CKPT_PATH` to the converted MaxText checkpoint path.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

> **Note (Gemma 4 26B specifics):**
>
> - Run the conversion with `scan_layers=False` so the resulting unscanned checkpoint matches the `scan_layers=False` setting used for the RL run below.
> - This RL recipe fine-tunes the **base** model (`google/gemma-4-26B-A4B`), not the instruction-tuned default. Pass `--hf_model_path=google/gemma-4-26B-A4B` explicitly when running `to_maxtext` — otherwise MaxText defaults to `HF_IDS[gemma4-26b]` = `google/gemma-4-26b-a4b-it`.

## Chat Template Configuration

Unlike an instruction-tuned tokenizer, the `google/gemma-4-26B-A4B` base tokenizer does **not** ship with a chat template, so the RL run must supply one explicitly. The [run_gemma4_26b_rl.sh](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/rl/scripts/run_gemma4_26b_rl.sh) script points at two files bundled with the repo (and therefore baked into your Docker image):

- `data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json` — a stripped-down data template that injects only the system prompt and question (no turn markers). This avoids the doubled `<start_of_turn>` delimiters that would occur if the default `gsm8k_rl.json` (which bakes literal Gemma turn markers into the message content) were combined with the tokenizer's Jinja chat template.
- `chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json` — the Gemma 3 chat template, wrapped in a JSON object with a `chat_template` key. It is applied by the tokenizer as the single source of truth for turn-marker formatting.

Both files are already included under `src/maxtext/examples/chat_templates/`, so no additional setup is required. If you want to customize the prompt formatting, edit these files (or point the config at your own) before building the Docker image.

## Run RL Workload

### Build and Upload MaxText Docker Image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](../build_maxtext.md).

### Submit your workload

```bash
# The Docker image you pushed in the previous step
export CLOUD_IMAGE_NAME=<IMAGE_NAME>
export DOCKER_IMAGE="gcr.io/${PROJECT_ID?}/${CLOUD_IMAGE_NAME?}"

# Run the RL training script on your cluster
run_tutorial maxtext/trainers/post_train/rl/scripts/run_gemma4_26b_rl.sh
```

> **Note:** The `run_gemma4_26b_rl.sh` script pins the Pathways component images to specific versions via the xpk `--server-image` and `--proxy-server-image` flags (set through the `PATHWAYS_SERVER_IMAGE` and `PATHWAYS_PROXY_SERVER_IMAGE` variables at the top of the script). The `--server-image` is used for both the Pathways resource-manager server and the workers (the reference config uses the same image for both). Update these variables if you need a different Pathways release.

### Monitor your workload

To monitor your job's progress, you can use `kubectl` to check the `Jobset` status and stream logs directly from the pods.

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name
kubectl get pods | grep ${WORKLOAD_NAME}

# stream the logs from the running pod (replace <POD_NAME> with the name you found)
kubectl logs -f <POD_NAME>
```

Alternatively, after running the bash script, you will also get a link to the Google Cloud Console to view your workload logs. Follow the link to view logs and monitor your workload's progress in the Cloud Console.

### Monitor RL Metrics

During RL training, you can monitor key metrics to track model convergence, reward trends, and hardware performance.

To enable Tunix-managed metrics measurement, set `enable_tunix_perf_metrics` to `true` in RL configurations. Note that this flag is already set to `True` by default for this tutorial workload. When enabled, Tunix automatically collects and uploads these metrics to TensorBoard.

For a complete list of collected metrics, see the [Tunix Metrics Documentation](https://tunix.readthedocs.io/en/latest/metrics.html). Key metrics to monitor include:

- **Model Quality & Reward Metrics:**
  - `rewards/mean`: The average reward across the batch (crucial for tracking learning progress).
  - `score/mean`: The average raw score from the reward model before applying the KL penalty.
- **Rollout & Generation Metrics:**
  - `rollout_time`: How long each rollout step takes.
  - `completions/mean_length`: The average token length of generated completions.
  - `actor_dequeue_time`: The time spent waiting for data from the rollout workers (relevant when async rollout is enabled).
- **Performance & Efficiency Metrics:**
  - `step_time_sec`: The execution time for a single training step.

## Convert Checkpoint to Hugging Face Format

Refer to [MaxText to Hugging Face](../../guides/checkpointing_solutions/convert_checkpoint.md#maxtext-to-hugging-face) to convert a MaxText checkpoint back to Hugging Face format. You can find an example script to convert the `gemma4-26b` model to Hugging Face format [here](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/gemma4/26b/test_gemma4_to_hf.sh).

> **Note (Gemma 4 26B specifics):** Because this recipe fine-tunes the **base** model, pass `--hf_model_path=google/gemma-4-26B-A4B` to `to_huggingface` so the exported checkpoint bundles the base tokenizer. Without it, `to_huggingface` sources the tokenizer from `HF_IDS[gemma4-26b]` = `google/gemma-4-26b-a4b-it` (the instruction-tuned model). Also keep `scan_layers=False`, matching the RL run that produced the checkpoint.
