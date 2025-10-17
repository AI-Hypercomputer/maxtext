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

# Try GRPO

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the Llama3.1 8B-IT model on the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can enhance your model's problem-solving skills on mathematical word problems, coding problems, etc.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group's performance to update the policy.

We use Tunix as the library for GRPO.
And we use vLLM as the library for efficient model inference and generation.

In this tutorial we use a single host TPUVM such as `v6e-8/v5p-8`. Let's get started!

## Setup your virtual environment

### Create a Python3.12 venv if not already pre-existing and install MaxText dependencies
```
bash setup.sh
```

### Activate your virtual environment (Skip if you have already done this for running `bash setup.sh` )
```
# Replace with your virtual environment name if not using this default name
venv_name="maxtext_venv"
source ~/$venv_name/bin/activate
```

## vLLM and tpu-commons installations

Next, run the following bash script to get all the necessary installations inside the virtual environment.
This will take few minutes. Follow along the installation logs and look out for any issues!

```
bash ~/maxtext/src/MaxText/examples/install_tunix_vllm_requirement.sh
```

1. It installs `pip install keyring keyrings.google-artifactregistry-auth` which enables pip to authenticate with Google Artifact Registry automatically.
2. Next, it installs `vLLM` for Jax and TPUs from the artifact registry `https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/`
3. Then, it installs `tpu-commons` from the same artifact registry.

`tpu_commons` is the TPU backend for vLLM. You will need both libraries to run vLLM on tpus.
We use the scheduler code from vLLM, and the model runner code from `tpu_commons`


## Run GRPO

Finally, run the script

`python ~/maxtext/src/MaxText/examples/grpo_llama3_1_8b_demo.py`

The overview of the demo script is as follows:

1. We load a policy model and a reference model. Both are copies of `Llama3.1-8b-Instruct`.
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark after the post-training with GRPO.
