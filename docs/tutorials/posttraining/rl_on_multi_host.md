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

# Reinforcement Learning on Multi-Host TPUs

This tutorial provides step-by-step instructions for setting up the environment
and training the Llama3.1 70B-IT model on the GSM8K math reasoning dataset using
[Pathways for orchestration](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)
on multi-host TPU-VMs, such as `v5p-128`.

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

## Table of Contents

- [Prerequisites](#prerequisites)
- [Build and Upload MaxText Docker Image](#build-and-upload-maxtext-docker-image)
- [Setup Environment Variables](#setup-environment-variables)
- [Get Your Model Checkpoint](#get-your-model-checkpoint)
- [Submit your RL workload via Pathways](#submit-your-rl-workload-via-pathways)
- [Monitor and clean up](#monitor-and-clean-up)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

- Access to a Google Cloud Project with TPU quotas.
- **IAM Roles** required:
  - **Kubernetes Engine Developer** (`roles/container.developer`) to submit and manage workloads on GKE.
  - **Artifact Registry Writer** (`roles/artifactregistry.writer`) to upload Docker images.
  - **Storage Admin** (`roles/storage.admin`) or **Storage Object Admin** (`roles/storage.objectAdmin`) combined with **Storage Legacy Bucket Reader** (`roles/storage.legacyBucketReader`) on your GCS bucket to read/write checkpoints and logs. (Note: A bucket-level read permission like `storage.buckets.get` is required by JAX/TensorStore to verify bucket existence and metadata; using `roles/storage.objectAdmin` alone will cause a misleading "bucket not found" error).
- A Hugging Face account with an access token for downloading models.
- Cluster Toolkit installed and configured. Follow [Running MaxText with Cluster Toolkit](../../run_maxtext/run_maxtext_via_cluster_toolkit.md) for `gcluster` setup.
  - **Important:** Modern GKE clusters require the GKE auth plugin. If you encounter `gke-gcloud-auth-plugin not found` when running `kubectl` commands, you must install it locally (e.g., `sudo apt-get install google-cloud-cli-gke-gcloud-auth-plugin` for `apt` installations, or `gcloud components install gke-gcloud-auth-plugin` for standalone archive installations).
- A Pathways-ready GKE cluster configured for Cluster Toolkit, with healthy Kueue and JobSet components (see [create a GKE cluster with Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) and [Cluster Toolkit documentation](https://cloud.google.com/cluster-toolkit/docs/overview)).
- **Docker** installed and configured for sudoless use. Follow the steps to [configure sudoless Docker](https://docs.docker.com/engine/install/linux-postinstall/).

## Build and upload MaxText Docker image

For instructions on building and uploading the MaxText Docker image with post-training dependencies, please refer to the [official documentation](build-docker).

## Setup Environment Variables

Set up the following environment variables to configure your training run. Replace
placeholders with your actual values.

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g. 'llama3.1-70b-Instruct' # replace with another model from src/maxtext/configs/types.py if needed

# Your Hugging Face access token. Required to download gated models like Llama.
# You can generate one at https://huggingface.co/settings/tokens.
export HF_TOKEN=<HF_TOKEN>

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Workload names cannot exceed 28 characters (or 22 characters when using Pathways due to Kubernetes 63-byte coordinator label limits) and must be valid DNS labels (lowercase alphanumeric and hyphens).
export RUN_NAME=<RUN_NAME>

# The directory containing the MaxText-compatible model checkpoint.
# If you are converting from a Hugging Face checkpoint, see:
# [Checkpoint Conversion Guide](../../guides/checkpointing_solutions/convert_checkpoint.md)
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY?}/${RUN_NAME?}/0/items

# -- Workload configuration --
# Your GCP project ID. Find it on the [Cloud Console Dashboard](https://console.cloud.google.com/home/dashboard).
# If you've already set it in your local config, you can retrieve it via:
# gcloud config get-value project
export PROJECT_ID=<PROJECT_ID>

# The GCP location (listed as "Location" in the UI) and name of your
# TPU-enabled GKE cluster. Both can be found on the
# [Cloud Console](https://console.cloud.google.com/kubernetes/list).
export LOCATION=<LOCATION> # e.g., 'us-central1' or 'us-central1-a'
export GKE_CLUSTER=<CLUSTER_NAME>

# Cluster Toolkit workload placement. Specify the compute type (machine type)
# and topology matching your TPU node pool.
# For example:
#   - v5p-128: COMPUTE_TYPE='ct5p-hightpu-4t', TOPOLOGY='4x4x4'
#   - v6e-256: COMPUTE_TYPE='ct6e-hightpu-4t', TOPOLOGY='16x16'
# To inspect the accelerator and topology labels on your GKE cluster nodes:
# kubectl get nodes -l cloud.google.com/gke-tpu-accelerator -o custom-columns=NAME:.metadata.name,ACCELERATOR:.metadata.labels.cloud\\.google\\.com/gke-tpu-accelerator,TOPOLOGY:.metadata.labels.cloud\\.google\\.com/gke-tpu-topology
export COMPUTE_TYPE=<COMPUTE_TYPE>
export TOPOLOGY=<TOPOLOGY>

# The Docker image you pushed in the previous step
export DOCKER_IMAGE=<IMAGE_NAME>
```

## Get Your Model Checkpoint

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the
following environment variable and move on to the next section.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting from a Hugging Face checkpoint

Refer to the steps in [Hugging Face to MaxText](../../guides/checkpointing_solutions/convert_checkpoint.md#hugging-face-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```bash
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

> [!IMPORTANT]
> **Automatic `scan_layers` Resolution:**
> MaxText automatically loads `scan_layers` from the checkpoint's saved metadata when resuming (via `load_parameters_path`) if you do not explicitly specify it on the command-line.
>
> - You do not need to manually supply `scan_layers=False` (or `scan_layers=True`) when loading checkpoints; MaxText will configure this automatically.
> - If you do explicitly provide a `scan_layers` argument, it must match the checkpoint's saved setting or a `ValueError` mismatch error will be raised.
>   See the [Checkpoints concept guide](../../reference/core_concepts/checkpoints.md) for more details.

## Submit your RL workload via Pathways

See the **Troubleshooting** section for concise instructions on how to retry or
resume a failed workload.

Configure `kubectl` and `gcluster` for the target cluster before submitting:

```bash
gcloud config set project ${PROJECT_ID?}
gcloud container clusters get-credentials ${GKE_CLUSTER?} \
  --location ${LOCATION?} \
  --project ${PROJECT_ID?}
gcluster job config set project ${PROJECT_ID?}
gcluster job config set cluster ${GKE_CLUSTER?}
gcluster job config set location ${LOCATION?}
```

### Submit GRPO workload

```bash
gcluster job submit \
  --image=${DOCKER_IMAGE?} \
  --name=${RUN_NAME?} \
  --pathways \
  --compute-type=${COMPUTE_TYPE?} \
  --topology=${TOPOLOGY?} \
  --num-slices=1 \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command="ENABLE_PATHWAYS_PERSISTENCE=1 \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  rollout_tensor_parallelism=8 \
  hf_access_token=${HF_TOKEN?} \
  enable_single_controller=True"
```

### Submit GSPO workload

```bash
gcluster job submit \
  --image=${DOCKER_IMAGE?} \
  --name=${RUN_NAME?} \
  --pathways \
  --compute-type=${COMPUTE_TYPE?} \
  --topology=${TOPOLOGY?} \
  --num-slices=1 \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command="ENABLE_PATHWAYS_PERSISTENCE=1 \
python3 -m maxtext.trainers.post_train.rl.train_rl \
  model_name=${MODEL?} \
  load_parameters_path=${MAXTEXT_CKPT_PATH?} \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  rollout_tensor_parallelism=8 \
  hf_access_token=${HF_TOKEN?} \
  rl.loss_algo=gspo-token \
  enable_single_controller=True"
```

## Profiling and ML Diagnostics

The RL trainer profiles through tunix's targeted RL profiler, which captures one
trace per role: `actor` (`update_actor`) and `rollout` (`generate`). The window
is counted in **role invocations**, not training steps (`update_actor` runs once
per micro-batch; eval actor passes share the actor counter):

```bash
profiler=xplane \
rl.profiler_start_invocation=2 \
rl.profiler_num_invocations=1
```

The step-window flags used by pretraining and SFT (`skip_first_n_steps_for_profiler`,
`profiler_steps`, `profile_periodically_period`, `upload_all_profiler_results`, ...)
are **not accepted** by `train_rl`; passing one fails at config time with the list
of valid fields.

- Without ML Diagnostics, traces are written to
  `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/tensorboard/plugins/profile/<role>_invocation_<n>/`.
- With `managed_mldiagnostics=true managed_mldiagnostics_on_demand_profiling=false`,
  MaxText creates one ML Diagnostics run (trainer config plus `sampler.<key>` for
  sampler-side differences), streams every scalar to **both** ML Diagnostics and
  TensorBoard, and uploads traces under
  `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/managed-mldiagnostics/<run id>/plugins/profile/`.
  The `<run id>` is assigned by the SDK (on GKE it is derived from
  `GKE_DIAGON_IDENTIFIER`, not from `run_name`); the pod log line
  `xprof initialized. Profiling output path set to: ...` prints it. The run only
  appears in the console when the launcher sets `GKE_DIAGON_IDENTIFIER` and
  `GKE_DIAGON_METADATA`.
- **Pathways traces cover worker 0 only by default.** On a disaggregated topology
  the vLLM sampler runs on another worker, so its TPU activity is missing from the
  `rollout` trace unless the launch command exports
  `PATHWAYS_MAX_NUM_HOSTS` (for example `PATHWAYS_MAX_NUM_HOSTS=1000` to trace
  every worker) before `python3 -m maxtext.trainers.post_train.rl.train_rl`.

## Monitor and clean up

```bash
gcluster job list
# Note: For Pathways workloads (> 5 pods), specify --main-only=false to retrieve logs from all pods:
gcluster job logs ${RUN_NAME?} --main-only=false
gcluster job cancel ${RUN_NAME?}
```

You can also inspect the Kubernetes resources directly:

```bash
kubectl get jobset -l gcluster.google.com/workload=${RUN_NAME?}
# In Pathways workloads, use the jobset-name label to select all pods (both pathways-head and worker pods):
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?}
kubectl logs -f -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?} --all-containers=true --max-log-requests=64
```

## Troubleshooting

- **Authentication Issues**: Ensure your `HF_TOKEN` environment variable is
  set correctly and has access to the required models.
- **Resource Quotas**: Verify you have sufficient TPU quotas in your GCP
  project.
- **Docker Build Failures**: Check that all dependencies are correctly
  installed and authentication is configured.
- **Workload Failures**: Review the logs for specific error messages and
  ensure all environment variables are properly set.
- **Parallelism ValueError (At most one can be -1)**: If you see `ValueError: At most one of rollout_tensor_parallelism, ... can be -1 (auto-derived)`, it means you have not explicitly defined the rollout parallelism parameters.
  - **Solution**: Explicitly pass at least one of them in your training command (e.g., `rollout_tensor_parallelism=8` as shown in the example commands above).
- **Workload retry / resume**:
  - **Retry (fresh run)**: Use a unique run name to avoid overwriting
    outputs:
    ```bash
    export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY?}/${RUN_NAME?}/0/items
    export RUN_NAME=${RUN_NAME?}-retry1
    ```
    Then submit the Cluster Toolkit workload. If a "workload already exists" error occurs, pick
    a new name or cancel the previous job (`gcluster job cancel ${RUN_NAME?}`).
  - **Resume from checkpoint**: Keep the same `RUN_NAME` and set the
    checkpoint path: `export load_parameters_path=${MAXTEXT_CKPT_PATH?}/checkpoint-0000`. Then submit
    the workload again.
  - **Tip**: Verify the checkpoint exists in GCS with read access before
    resuming.

For more detailed troubleshooting, refer to the
[MaxText documentation](../../index.md) and
[Cluster Toolkit guide](../../run_maxtext/run_maxtext_via_cluster_toolkit.md).
