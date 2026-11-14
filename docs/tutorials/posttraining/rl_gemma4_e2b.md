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

# Reinforcement Learning with gemma4-e2b on Multi-Host TPUs

This tutorial provides step-by-step instructions for training the gemma4-e2b model with GRPO on the
[OpenMathInstruct-2 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) on a Cloud TPU v7x (Ironwood)
GKE cluster using a `tpu7x` 4x4x4 slice (64 chips) with Cluster Toolkit. The rollout engine is vLLM on TPU
running the MaxText model (`MaxTextForCausalLM`), and responses may be up to 8192 tokens long.

> **Note:** gemma4-e2b shares the key/value caches of its last 20 layers with earlier layers (KV sharing). Rollouts
> through the MaxText vLLM adapter require the KV-sharing write-back fix in
> `maxtext/integration/vllm/hybrid_cache_utils.py` (`scatter_layer_kv_caches` must not write untouched alias
> entries back over the donor caches). Without it, generations degenerate into repeated tokens. Use a MaxText
> version that includes it.

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU v7x quota for a 4x4x4 slice.
- A Hugging Face account with an access token for downloading models (the `google/gemma-4-E2B-it` repository is gated; request access before proceeding).
- Permissions for Google Artifact Registry (Artifact Registry Writer role).
- Cluster Toolkit installed and configured. Follow [Running MaxText with Cluster Toolkit](../../run_maxtext/run_maxtext_via_cluster_toolkit.md) for `gcluster` setup.
- A Pathways-ready GKE cluster configured for Cluster Toolkit, including healthy Kueue and JobSet components (see [create a GKE cluster with Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) and [Cluster Toolkit documentation](https://cloud.google.com/cluster-toolkit/docs/overview)).
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

# An arbitrary string to identify this specific run.
export RUN_NAME=<RUN_NAME>

# Number of GRPO updates (default 500; use 20 for a quick performance check).
export NUM_BATCHES=500
```

## Authenticate with Hugging Face

`google/gemma-4-E2B-it` is a gated repository. Log in on the machine where you convert the checkpoint, and export a
token for the training job (the script passes it to the container as `hf_access_token`):

```bash
hf auth login
export HF_TOKEN=<YOUR_HF_TOKEN>
```

## Get Your MaxText Compatible Model Checkpoint

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

> **Note (Gemma 4 E2B specifics):**
>
> - Run the conversion with `scan_layers=False` (the `gemma4_small` decoder block is incompatible with `nn.scan`); the resulting unscanned checkpoint matches the `scan_layers=False` setting used for the RL run below.
> - This recipe fine-tunes the **instruction-tuned** model, `google/gemma-4-E2B-it`, which is the default `HF_IDS[gemma4-e2b]` for `to_maxtext`.

## Chat Template Configuration

The [run_gemma4_e2b_rl.sh](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh) script passes `data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json`, bundled with the repo (and therefore baked into your Docker image). It injects only the system prompt and the question, with no turn markers.

The turn markers come from the chat template that ships with the `google/gemma-4-E2B-it` tokenizer (`<|turn>user … <turn|>`), so no `chat_template_path` is needed. MaxText only falls back to `chat_template` / `chat_template_path` when the tokenizer has none, as with the base `google/gemma-4-E4B` tokenizer in the gemma4-e4b recipe.

## Run RL Workload

### Build and Upload MaxText Docker Image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](build-docker).

### Submit your workload

```bash
# The Docker image you pushed in the previous step
export DOCKER_IMAGE=<IMAGE_NAME>

# Run the RL training script on your cluster
run_tutorial maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh
```

> **Note:** The script pins the Pathways component images (`PATHWAYS_SERVER_IMAGE`, `PATHWAYS_PROXY_SERVER_IMAGE`) to the release it was validated with (`jax_0.11.0`, matching the jax in the MaxText image). Update these variables if you need a different Pathways release.

### What the configuration does

| setting             | value                                                                                       | why                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| slice               | `tpu7x-standard-4t`, `4x4x4` (64 chips, 128 devices)                                        | trainer and vLLM rollout share the slice (`trainer_devices_fraction=sampler_devices_fraction=1.0`)                                                                               |
| batch               | `batch_size=256`, `rl.num_generations=8`                                                    | 2048 completions per GRPO update                                                                                                                                                 |
| lengths             | `max_prefill_predict_length=4096`, `max_target_length=12288`                                | responses up to 8192 tokens, so rollouts are not truncated in practice (≈0.1 % hit the cap)                                                                                      |
| trainer packing     | `max_seq_token_per_tpu=12288`                                                               | packs completions into rows of 12288 tokens; an unpacked micro-batch of 64 rows does not shard over 128 devices                                                                  |
| rollout parallelism | `rollout_data_parallelism=128`, `rollout_tensor_parallelism=1`                              | one vLLM data-parallel rank per device; gemma4-e2b fits on one chip                                                                                                              |
| continue-decode     | `enable_continue_decode`, `max_decode_steps=1024`, `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`   | runs each 1024-step decode window on device to its full depth instead of returning to the host whenever a sequence ends (≈ 30 % faster updates on this workload)                 |
| scheduling          | `DP_SCHED_ROUTING=round_robin`                                                              | assigns new requests to the 128 data-parallel ranks round-robin instead of querying every rank's load per request (tpu-inference 2026-09-18 or newer, per the post-training pin) |
| KV cache            | `hbm_utilization_vllm=0.3`, `vllm_block_size=128`, `free_kv_cache_during_weight_sync=False` | keeps the KV cache allocated across weight syncs                                                                                                                                 |
| attention           | `attention=flash`, explicit `local_sa_*` block sizes and layouts                            | gemma4's sliding-window layers use the local splash-attention blocks                                                                                                             |
| power               | `--xla_tpu_dvfs_p_state=6`                                                                  | DVFS state used for the reference numbers below                                                                                                                                  |

### Monitor your workload

To monitor your job's progress, you can use `gcluster` or `kubectl` to check the `JobSet` status and stream logs directly:

```bash
# Check job status with Cluster Toolkit
gcluster job list

# Stream logs with Cluster Toolkit (specify --main-only=false for Pathways workloads)
gcluster job logs ${RUN_NAME?} --main-only=false

# Alternatively, check JobSet status with kubectl
kubectl get jobset -l gcluster.google.com/workload=${RUN_NAME?}

# List pods (use jobset-name to select both head and worker pods in Pathways)
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?}

# Stream logs with kubectl
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?} --all-containers=true --max-log-requests=64
```

### Monitor RL Metrics

`enable_tunix_perf_metrics=True` is set in the script, so Tunix collects metrics and uploads them to TensorBoard
under `BASE_OUTPUT_DIRECTORY`. See the [Tunix Metrics Documentation](https://tunix.readthedocs.io/en/latest/metrics.html)
for the full list; the most useful are `rewards/mean`, `completions/mean_length`, `rollout_time` and `step_time_sec`.

### Reference performance

Measured with this configuration on a TPU v7x 4x4x4 slice (MaxText image of 2026-09-22 with the KV-sharing fix,
tpu-inference at its 2026-09-21 head, older than the commit now pinned in
`src/dependencies/extra_deps/post_train_github_deps.txt`,
Pathways `jax_0.11.0`, `google/gemma-4-E2B-it` checkpoint, 8 updates). The first update includes compilation, and the
time per update depends on how long the sampled completions are.

| update  | 2    | 3    | 4    | 5    | 6    | 7    |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- |
| seconds | 54.4 | 54.5 | 56.5 | 56.2 | 67.4 | 58.2 |

Median: about 56 s per GRPO update (2048 completions with responses up to 8192 tokens). For reference, the same
workload with verl on 64 GPUs runs at 31.7–35.0 s per step.

## Convert Checkpoint to Hugging Face Format

Refer to [MaxText to Hugging Face](maxtext-to-hf) to convert a MaxText checkpoint back to Hugging Face format. Keep
`scan_layers=False`, since the `gemma4_small` decoder block is not compatible with scanned layers.
