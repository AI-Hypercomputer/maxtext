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

(getting-started)=

# Getting Started

Welcome to MaxText! This guide will help you get started with running your first MaxText workloads. Whether you are working on a single host or scaling up to a multihost environment using Cloud TPUs or NVIDIA GPUs, this page provides the starting point for your journey. Follow the steps below to install MaxText, train your first model, and run inference.

## Prerequisites

1. To store logs and checkpoints, [create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) in your project. To run MaxText, the TPU or GPU VMs must have read/write permissions for the bucket. These permissions are granted by service account roles, such as the `STORAGE ADMIN` role.

2. MaxText reads a yaml file for configuration. We also recommend reviewing the configurable options in `configs/base.yml`. This file includes a decoder-only model of ~1B parameters. The configurable options can be overwritten from the command line. For instance, you can change the `steps` or `log_period` by either modifying `configs/base.yml` or by passing in `steps` and `log_period` as additional arguments to the `train.py` call. Set `base_output_directory` to a folder in the bucket you just created.

3. **Checkpoint Conversion**: In order to run MaxText on HuggingFace checkpoints, you must convert them to the MaxText/Orbax format first. For detailed instructions, see the [Checkpoint Conversion Guide](guides/checkpointing_solutions/convert_checkpoint.md).

## Running MaxText on a Single Host

This procedure describes how to run MaxText on a single GPU or TPU host.

### 1. Installation

Before running MaxText, you must install it on your VM.

- For detailed installation instructions, see the [Installation Guide](install_maxtext.md).
- For TPU VMs, install `maxtext[tpu]` for pre-training, or `maxtext[tpu-post-train]` for post-training.
- For GPU VMs, ensure you install `maxtext[cuda12]`.

### 2. Running Pre-training

To get started with training your first model, refer to the [Pre-training Tutorial](tutorials/pretraining.md).

### 3. Running Post-training

To fine-tune your model or apply post-training techniques (such as SFT or RL), refer to the [Post-training Tutorial](tutorials/post_training_index.md). This guide covers various post-training workflows.

### 4. Running Inference

To run inference (decoding) using MaxText models, refer to the [Inference Tutorial](tutorials/inference.md). This guide covers offline and online inference, as well as integration with vLLM.

## Running MaxText on Multiple Hosts

Google Kubernetes Engine (GKE) is the recommended way to run MaxText on multiple hosts. It provides a managed environment for deploying and scaling containerized applications, including those that require TPUs or GPUs. See [Running Maxtext with XPK](run_maxtext/run_maxtext_via_xpk.md) for details.

## Running MaxText in Notebooks

You can run MaxText interactively using Jupyter notebooks, Google Colab, or Visual Studio Code. Refer to the [Notebook Guide](guides/run_python_notebook.md) for instructions on setting up your notebook environment on TPUs.

## Next steps: preflight optimizations

After you get workloads running, there are optimizations you can apply to improve performance. For more information, see [Optimization Tips](guides/optimization.md).
