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

This guide is the Cluster Toolkit replacement for the former XPK/Pathways
workflow. It runs MaxText as a multi-host GKE JobSet using Cluster Toolkit's
`gcluster` CLI.

Cluster Toolkit does not expose a direct replacement for the Pathways proxy or
headless mode. The supported replacement is a workload that runs entirely in
the GKE cluster. The former Pathways proxy and headless instructions are not
part of this guide because they cannot be reproduced with `gcloud` or the
standard `gcluster job submit` workflow.

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
```

Verify the cluster prerequisites before submitting a multi-host job:

```bash
kubectl get nodes
kubectl get crd jobsets.jobset.x-k8s.io
kubectl get crd clusterqueues.kueue.x-k8s.io
```

## Configure the workload

```bash
export RUN_NAME="maxtext-run-$(date +%Y%m%d-%H%M%S)"
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET_PATH>
export COMPUTE_TYPE=<CLUSTER_TOOLKIT_COMPUTE_TYPE>
export TOPOLOGY=<TPU_TOPOLOGY>
export BASE_IMAGE=<ARTIFACT_REGISTRY_BASE_IMAGE>
```

For example, `ct5p-hightpu-4t` with `4x4x4` is a multi-host TPU topology.
Choose a compute type and topology supported by the target cluster.

## Run a batch workload

The following command runs the same synthetic MaxText batch task previously
shown in the XPK/Pathways guide, using a standard GKE JobSet:

```bash
gcluster job submit \
  --base-image ${BASE_IMAGE?} \
  --build-context . \
  --command "python3 -m maxtext.trainers.pre_train.train \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    dataset_type=synthetic \
    run_name=${RUN_NAME?}" \
  --name ${RUN_NAME?} \
  --compute-type ${COMPUTE_TYPE?} \
  --topology ${TOPOLOGY?}
```

To run a real workload, replace the command arguments with the arguments for
your model, dataset, checkpoint, and training schedule. The image-based form
can be used when the image is already built:

```bash
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
gcluster job logs ${RUN_NAME?}
gcluster job cancel ${RUN_NAME?}
```

You can also inspect the Kubernetes resources directly:

```bash
kubectl get jobset -l gcluster.google.com/workload=${RUN_NAME?}
kubectl get pods -l gcluster.google.com/workload=${RUN_NAME?}
```

## Compatibility note

The former Pathways workflow provided a proxy backend and a headless mode that
ran the Python process outside the workload container. Cluster Toolkit's
standard `gcluster job submit` workflow does not provide those services. Use
the workload form above for new deployments and remove Pathways-specific JAX
settings such as `JAX_BACKEND_TARGET=grpc://127.0.0.1:29000`.
