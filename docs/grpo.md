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

# Try GRPO!

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the Llama3.1 8B-IT model on the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can enhance your model's problem-solving skills on mathematical word problems, coding problems, etc.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by eliminating the need for a separate value function model. GRPO works by generating multiple responses for a given prompt, evaluating these responses using a reward model, and then calculating a relative advantage based on the group's performance to update the policy.

We use Tunix as the library for GRPO. 
And we use vLLM as the library for efficient model inference and generation.
 
In this tutorial we use a single host TPUVM such as `v6e-8/v5p-8`. Let's get started!

## Setup your virtual environment

### Create a virtual environment

```
# Install uv using the standalone installer (doesn't require apt/dpkg)
curl -LsSf https://astral.sh/uv/install.sh | sh

# add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install Python 3.12 
uv python install 3.12

# Verify Python 3.12 is installed 
uv python list 

cd ~

# Create a virtual environment with Python 3.12
uv venv .venv-demo-py312  --python 3.12
```

### Activate your virtual environment
```
source ~/.venv-demo-py312/bin/activate
```

## Installations

Next, run the following bash script to get all the necessary installations inside the virtual environment.
This will take several minutes. Follow along the installation logs and look out for any issues!

```
bash ~/maxtext/MaxText/examples/install_tunix_vllm_requirement.sh 
```

1. It runs `gcloud auth login` - This command authenticates you, the user, with Google Cloud.
2. Next, it runs `cloud auth application-default login` - This command sets up authentication for applications and libraries running on your local machine.
3. Then, it runs `pip install keyring keyrings.google-artifactregistry-auth` - This command installs Python packages that enable pip to authenticate with Google Artifact Registry automatically.
4. It run `bash setup.sh` thereafter, which installs all the dependencies of MaxText. 
5. Next, it installs `vLLM` for Jax and TPUs from the artifact registry `https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/`
6. Then, it installs `tpu-commons` from the same artifact registry. 

`tpu_commons` is the TPU backend for vLLM. You will need both libraries to run vLLM on tpus.
We use the scheduler code from vLLM, and the model runner code from `tpu_commons`


## Run MaxText

Finally, run the script

`python ~/maxtext/MaxText/examples/grpo_llama3_demo.py`

The overview of the demo script is as follows:

1. We load a policy model and a reference model. Both are copies of `Llama3.1-8b-Instruct`.
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark after the post-training with GRPO.
