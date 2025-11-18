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

## Create virtual environment and Install MaxText dependencies
Follow instructions in [Install MaxText](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/install_maxtext.md), but 
recommend creating the virtual environment outside the `maxtext` directory.

## vLLM and tpu-inference installations

Next, run the following bash script to get all the necessary installations inside the virtual environment (for e.g., `maxtext_venv`).
This will take few minutes. Follow along the installation logs and look out for any issues!

```
bash ~/maxtext/src/MaxText/examples/install_tunix_vllm_requirement.sh
```

Primarily, it installs `vllm-tpu` which is [vllm](https://github.com/vllm-project/vllm) and [tpu-inference](https://github.com/vllm-project/tpu-inference) and thereby providing TPU inference for vLLM, with unified JAX and PyTorch support.


## Run GRPO

Finally, run the command

```
python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml \
  model_name=llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=$WORKLOAD \
  base_output_directory=$OUTPUT_PATH \
  hf_access_token=$HF_TOKEN
```

The overview of the what this run will do is as follows:

1. We load a policy model and a reference model. Both are copies of `Llama3.1-8b-Instruct`.
2. Evaluate the policy model's performance on GSM8K math reasoning benchmark.
3. Train the policy model using GRPO.
4. Evaluate the policy model's performance on GSM8K math reasoning benchmark after the post-training with GRPO.
