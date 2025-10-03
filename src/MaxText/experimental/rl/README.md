# Group Relative Policy Optimization (GRPO) in MaxText

This directory contains code and documentation for **GRPO**, a reinforcement learning algorithm designed to optimize language model policies within the MaxText framework. GRPO enables training language models to perform specific tasks by optimizing for a reward signal, going beyond standard language modeling objectives. This implementation leverages techniques such as **Fully Sharded Data Parallelism (FSDP)** for training and **Data Parallelism (DP) + Tensor Parallelism (TP)** for inference.

## Key Concepts

*   **GRPO (Group Relative Policy Optimization):** A policy optimization algorithm that uses reinforcement learning principles to improve the quality of a language model's responses based on a provided reward signal.
*   **FSDP (Fully Sharded Data Parallelism):** A technique for distributing model parameters across multiple devices during training, allowing for efficient training of large models.
*   **DP (Data Parallelism):** A technique for distributing data across multiple devices during inference, allowing for efficient inference of large models.
*   **Sharding:** For more information on sharding, refer to [getting_started/Sharding.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/Sharding.md)
*   **[Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro):** A system that allows to orchestrate computations across multiple TPU slices, enabling training and inference on very large TPU topologies.
*   **inference_rollouts:** This is a config parameter to control the frequency in training steps to reshard and copy training parameters to the inference mesh. Must be an integer between 1 and `steps` (total number of training steps).

## Getting Started

### Prerequisites

1.  **Google Cloud Platform (GCP):** You need a GCP project with sufficient TPU quota.
2.  **GKE Cluster with Pathways:**
    *   Follow the instructions to create a GKE cluster with Pathways support: [Create a GKE Cluster with Pathways](https://cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)
    *   This involves setting up node pools for both training and inference.
3.  **Pathways Container Images:** Your GCP service account must be allowlisted to access the Pathways container images. Contact your Google Cloud representative for assistance.
4.  **XPK:** (Recommended) Install and configure XPK, the Accelerated Processing Kit, to simplify the management of your GKE cluster and Pathways environment.
5. **HF_TOKEN:** You need to have a Hugging Face token to access the models.

### Setup

*   Make sure your GKE cluster is properly configured with Pathways. This includes having the correct node pools for training and inference, and that the Pathways Resource Manager, Proxy, and Worker are correctly deployed.
*   Set the environment variables for your cluster, project, and zone.

## Running GRPO

This repository includes a shell script, `end_to_end/tpu/test_grpo.sh`, that demonstrates how to run GRPO on a v5p-256 cluster.

**How it works:**

*   **TPU Topology:** This script targets a v5p-256 cluster, which consists of 256 TPU cores.
*   **Device Allocation:**
    *   **Training:** 64 devices are reserved for training. The model is sharded using FSDP across these devices.
    *   **Inference (Sampling):** The other 64 devices are used for inference. The model is sharded using DP across samplers and TP within each sampler.
* **Samplers:** The inference devices are grouped into samplers.
*   **Example Command:**

```bash
HF_TOKEN=${HF_TOKEN} \
MODEL=llama3.3-70b \
TOKENIZER=meta-llama/Llama-3.3-70B-Instruct \
NUM_SAMPLERS=8 \
DEVICES_PER_SAMPLER=8 \
TRAINING_PER_DEVICE_BATCH_SIZE=1 \
INFERENCE_PER_DEVICE_BATCH_SIZE=8 \
STEPS=20 \
bash end_to_end/tpu/test_grpo.sh
