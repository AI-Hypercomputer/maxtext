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

# GRPO on single-host TPUs

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the Llama3.1 8B-IT model on the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can enhance your model's problem-solving skills on mathematical word problems, coding problems, etc.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group's performance to update the policy.

We use Tunix as the library for GRPO/GSPO.
And we use vLLM as the library for efficient model inference and generation.

In this tutorial we use a single host TPUVM such as `v6e-8/v5p-8`. Let's get started!

## Create virtual environment and Install MaxText dependencies
If you have already completed the [MaxText installation](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/install_maxtext.md), you can skip to the next section for vLLM and tpu-inference installations. Otherwise, please install MaxText using the following commands before proceeding.
```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Create virtual environment
export VENV_NAME=<your virtual env name> # e.g., maxtext_venv
pip install uv
uv venv --python 3.12 --seed $VENV_NAME
source $VENV_NAME/bin/activate

# 3. Install dependencies in editable mode
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps
```

## vLLM and tpu-inference installations

### From PyPI releases

Next, run the following bash script to get all the necessary installations inside the virtual environment (for e.g., `maxtext_venv`).
This will take few minutes. Follow along the installation logs and look out for any issues!

```
bash ~/maxtext/src/MaxText/examples/install_tunix_vllm_requirement.sh
```

Primarily, it installs `vllm-tpu` which is [vllm](https://github.com/vllm-project/vllm) and [tpu-inference](https://github.com/vllm-project/tpu-inference) and thereby providing TPU inference for vLLM, with unified JAX and PyTorch support.

### From Github

You can also locally git clone [tunix](https://github.com/google/tunix) and install using the instructions [here](https://github.com/google/tunix?tab=readme-ov-file#installation). Similarly install [vllm](https://github.com/vllm-project/vllm) and [tpu-inference](https://github.com/vllm-project/tpu-inference) from source following the instructions [here](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/#install-from-source)

## Setup the following environment variables before running GRPO

Setup following environment variables before running GRPO

```bash
# -- Model configuration --
export HF_MODEL='llama3.1-8b-Instruct'
export MODEL='llama3.1-8b'
export TOKENIZER='meta-llama/Llama-3.1-8B-Instruct'
export HF_TOKEN=<Hugging Face access token>

# -- MaxText configuration --
export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory

export RUN_NAME=<name for this run> # e.g., $(date +%Y-%m-%d-%H-%M-%S)
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/0/items
```

## Get your model checkpoint

You can convert a Hugging Face checkpoint to MaxText format using the `src/MaxText/utils/ckpt_conversion/to_maxtext.py` script. This is useful if you have a pre-trained model from Hugging Face that you want to use with MaxText.

First, ensure you have the necessary dependencies installed. Then, run the conversion script on a CPU machine. For large models, it is recommended to use the --lazy_load_tensors flag to reduce memory usage during conversion. This command will download the Hugging Face model and convert it to the MaxText format, saving it to the specified GCS bucket.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MAXTEXT_CKPT_PATH} \
    scan_layers=True hardware=cpu skip_jax_distributed_system=true

# Example of converting Llama3.1-70B using --lazy_load_tensor=true which uses around 86GB of RAM

python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=llama3.1-70b \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME} \
    scan_layers=True \
    hardware=cpu skip_jax_distributed_system=true \
    --lazy_load_tensors=true
```



## Run GRPO

Finally, run the command

```
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=${MODEL} \
  tokenizer_path=${TOKENIZER} \
  load_parameters_path=${MAXTEXT_CKPT_PATH} \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  hf_access_token=${HF_TOKEN}
```

The overview of the what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of `Llama3.1-8b-Instruct`.
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark after the post-training with GRPO.

GSPO (Group Sequence Policy Optimization)
MaxText can also run the GSPO variant by setting `loss_algo=gspo-token` when invoking `train_rl.py` (or when constructing the pyconfig argv list). 

## Run GSPO

Finally, run the command

```
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=$WORKLOAD \
  base_output_directory=$OUTPUT_PATH \
  hf_access_token=$HF_TOKEN \
  loss_algo=gspo-token
```

