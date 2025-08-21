# Objective
This guide provides comprehensive instructions for setting up MaxText on a local machine or single-host environment, covering everything from cloning the repo and dependency installation to building with Docker. By walking through the process of pre-training a small model, you will gain the foundational knowledge to run jobs on TPUs/GPUs.

---

# Prerequisites
Before you can begin a training run, you need to configure your storage environment and set up the basic MaxText configuration.

## Setup Google Cloud Storage Bucket
You'll need a GCS bucket to store all your training artifacts, such as logs, metrics, and model checkpoints.

1.  In your Google Cloud project, create a new storage bucket.
2.  Your TPU or GPU VMs require read/write access to this bucket. The simplest way to grant this is by assigning the `Storage Admin` (`roles/storage.admin`) role to the service account associated with your VMs.

## Setup Maxtext
MaxText uses a primary YAML file, `configs/base.yml`, to manage its settings. This default configuration sets up a llama2 style decoder-only model with approximately 1 billion parameters.

* Before running your first model, take a moment to review this file. Pay special attention to these core settings:
  - `run_name`: The name for your experiment.
  - `per_device_batch_size`: Controls how many examples are processed per chip. You may need to lower this for larger models to avoid running out of memory.
  - `max_target_length`: The maximum sequence length for the model.
  - `learning_rate`: The core hyperparameter for the optimizer.
  - Mode shape parameters: `base_num_decoder_layers`, `base_emb_dim`, `base_num_query_heads`, `base_num_kv_heads`, and `head_dim`.
* **Override Settings (Optional):** You can modify training parameters in two ways: by editing `configs/base.yml` directly or by passing them as command-line arguments to the training script which is the recommended method. For example, to change the number of training steps, you can pass `--steps=500` when running `train.py`.
* **Note**: You **must** update the variable `base_output_directory` which is initialized in `configs/base.yml` to point to a folder within the GCS bucket you just created (e.g., `gs://your-bucket-name/maxtext-output`).

---

# Development
Local development on a single host TPU/GPU VM is a convenient way to run MaxText on a single host. It doesn't scale to multiple hosts but is a good way to learn about MaxText. The following describes how to run Maxtext on TPU/GPU VMs.

## Run Maxtext on Single Host VM
1.  Create and SSH to the single host VM of your choice. You can use any available single host TPU, such as `v5litepod-8`, `v5p-8`, or `v4-8`. For GPUs, you can use `nvidia-h100-mega-80gb`, `nvidia-h200-141gb`, or `nvidia-b200`. For setting up a TPU VM, use the Cloud TPU documentation available at https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm. For a GPU setup, refer to the guide at https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus. 

2.  Clone MaxText onto that VM.
    ```bash
    # Clone the repository
    git clone https://github.com/google/maxtext.git
    cd maxtext
    ```

3.  Once you have cloned the repository, you have two primary options for setting up the necessary dependencies on your VM: Installing in a Python Environment, or building a Docker container. For single host workloads, we recommend to install dependencies in a python environment, and for multihost workloads we recommend the containerized approach.

Within the root directory of the cloned repo, install dependencies and the pre-commit hook by running:

```bash
# Create a virtual environment and install dependencies
python3.12 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate
bash setup.sh DEVICE={tpu|gpu}
```

### Run a Test Training Job
After the installation is complete, run a short training job using synthetic data to confirm everything is working correctly. This command trains a model for just 10 steps. Remember to replace `$YOUR_JOB_NAME` with a unique name for your run and `gs://<my-bucket>` with the path to the GCS bucket you configured in the prerequisites.

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```

**Optional**: If you want to try training on a real dataset, see [Data Input Pipeline](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline.md) for data input options from sources like HuggingFace, Grain, and TFDS.

### Generate Sample Output (Decoding)

To demonstrate model output, run the following command:

```bash
python3 -m MaxText.decode MaxText/configs/base.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  per_device_batch_size=1
```

**Note:** Because the model hasn't been properly trained, the output text will be random. To generate meaningful output, you need to load a trained checkpoint using the `load_parameters_path` argument.

## Running Models Using Provided Configs
Maxtext provides many OSS model configs that you can use directly to run training jobs on those model-specific architectures. These model-specific YAML files are located in `MaxText/configs/models` for TPU-oriented defaults, and `MaxText/configs/models/gpu` for GPU-oriented defaults.

### Training on TPUs
To use a pre-configured model for TPUs, you override the `model_name` parameter, and MaxText will automatically load the corresponding configuration from the `MaxText/configs/models` directory and merge it with the settings from `MaxText/configs/base.yml`.

<details open>
<summary><strong>llama3-8b (TPU)</strong></summary>

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
  model_name=llama3-8b \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
</details>

<details open>
<summary><strong>qwen3-4b (TPU)</strong></summary>

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
  model_name=qwen3-4b \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
</details>

### Training on GPUs
To use a GPU-optimized configuration, you should specify the path to the model's YAML file within the `MaxText/configs/models/gpu` directory as the main config file in the command. These files typically inherit from `base.yml` and set the appropriate `model_name` internally, as well as GPU-specific settings.

<details open>
<summary><strong>mixtral-8x7b (GPU)</strong></summary>

```bash
python3 -m MaxText.train MaxText/configs/models/gpu/mixtral_8x7b.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
This will load `gpu/mixtral_8x7b.yml`, which inherits from `base.yml`.
</details>

<details open>
<summary><strong>llama3-8b (GPU)</strong></summary>

```bash
python3 -m MaxText.train MaxText/configs/models/gpu/llama3-8b.yml \
  run_name=$YOUR_JOB_NAME \
  base_output_directory=gs://<my-bucket> \
  dataset_type=synthetic \
  steps=10
```
</details>

