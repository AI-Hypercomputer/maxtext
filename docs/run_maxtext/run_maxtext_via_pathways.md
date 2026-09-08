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

(run-pathways)=

# Via Pathways

This guide provides the Cluster Toolkit replacement for the former XPK/Pathways
workflow. It runs MaxText as a multi-slice Pathways workload on Google Kubernetes
Engine (GKE) using Cluster Toolkit's `gcluster` CLI with the `--pathways` flag.

Pathways workloads run entirely within the GKE cluster using `gcluster job submit --pathways`
with `enable_single_controller=True`. The Cluster Toolkit automatically configures the
Pathways head components (resource manager, proxy, and leader pod) on CPU node pools
and manages TPU slices as coordinated JobSets.

## Prerequisites

- A GCP project with TPU quota and a GKE cluster with the required TPU node
  pools.
- `gcloud`, `kubectl`, the GKE auth plugin, and `gcluster` installed. See
  [At scale with Cluster Toolkit](run_maxtext_via_cluster_toolkit.md).
- Kueue and JobSet installed and healthy on the GKE cluster.
- A MaxText image in Artifact Registry, or a Cluster Toolkit base image and a
  local MaxText checkout.

## Configure the cluster

Set the project and authenticate `kubectl` to the target cluster:

```bash
export PROJECT_ID=<GCP_PROJECT_ID>
export GKE_CLUSTER=<GKE_CLUSTER_NAME>
export ZONE=<GCP_ZONE>

gcloud config set project ${PROJECT_ID?}
gcloud container clusters get-credentials ${GKE_CLUSTER?} \
  --zone ${ZONE?} \
  --project ${PROJECT_ID?}

gcluster job config set project ${PROJECT_ID?}
gcluster job config set cluster ${GKE_CLUSTER?}
gcluster job config set location ${ZONE?}

# If using --base-image with Crane, ensure Docker credentials are configured for Artifact Registry:
gcloud auth configure-docker <REGION>-docker.pkg.dev
```

Verify the cluster prerequisites before submitting a multi-host job:

```bash
kubectl get nodes
kubectl get crd jobsets.jobset.x-k8s.io
kubectl get crd clusterqueues.kueue.x-k8s.io
```

## Configure the workload

```bash
# Note: Workload name cannot exceed 22 characters for Pathways workloads due to Kubernetes 63-byte label limit on coordinator address (`<name>-pathways-head-0-0.<name>`).
export RUN_NAME="pw-$(date +%m%d%H%M%S)"
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET_PATH>
export COMPUTE_TYPE=<CLUSTER_TOOLKIT_COMPUTE_TYPE>
export TOPOLOGY=<TPU_TOPOLOGY>
export BASE_IMAGE=<ARTIFACT_REGISTRY_BASE_IMAGE>
export NUM_SLICES=1
```

For example, `ct5p-hightpu-4t` with `4x4x4` is a multi-host TPU topology.
Choose a compute type and topology supported by the target cluster.

## Run a batch workload

The following command runs a synthetic MaxText batch training task using Pathways
orchestration via `gcluster job submit`:

```bash
gcluster job submit \
  --base-image ${BASE_IMAGE?} \
  --build-context . \
  --name ${RUN_NAME?} \
  --pathways \
  --compute-type ${COMPUTE_TYPE?} \
  --topology ${TOPOLOGY?} \
  --num-slices=${NUM_SLICES:-1} \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command "python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    dataset_type=synthetic \
    steps=10 \
    enable_single_controller=True \
    run_name=${RUN_NAME?}"
```

To run a real workload, replace the command arguments with the arguments for
your model, dataset, checkpoint, and training schedule. If you already have a
pre-built container image in Artifact Registry, pass `--image` directly instead
of `--base-image` and `--build-context`:

```bash
# Pathways multi-slice JobSet submission with pre-built image
gcluster job submit \
  --image <FULL_ARTIFACT_REGISTRY_IMAGE_URI> \
  --name ${RUN_NAME?} \
  --pathways \
  --compute-type ${COMPUTE_TYPE?} \
  --topology ${TOPOLOGY?} \
  --num-slices=${NUM_SLICES:-1} \
  --pathways-gcs-location=${BASE_OUTPUT_DIRECTORY?} \
  --command "python3 -m maxtext.trainers.pre_train.train <MAXTEXT_ARGS> enable_single_controller=True"

# Standard multi-host JobSet submission (without Pathways)
gcluster job submit \
  --image <FULL_ARTIFACT_REGISTRY_IMAGE_URI> \
  --command "python3 -m maxtext.trainers.pre_train.train <MAXTEXT_ARGS>" \
  --name ${RUN_NAME?} \
  --compute-type ${COMPUTE_TYPE?} \
  --topology ${TOPOLOGY?}
```

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
# Or view the MaxText training logs directly from the head container:
kubectl logs -l jobset.sigs.k8s.io/jobset-name=${RUN_NAME?} -c workload-container -f
```

## Compatibility note

The former XPK-based Pathways workflow supported an interactive proxy backend and headless mode where the Python process ran outside the workload container. In Cluster Toolkit, Pathways jobs run entirely within the GKE cluster using `gcluster job submit --pathways` with `enable_single_controller=True`. Interactive external proxy workflows (`JAX_BACKEND_TARGET=grpc://127.0.0.1:29000`) are not used with Cluster Toolkit; the entire controller and training loop run inside the cluster workload container.
