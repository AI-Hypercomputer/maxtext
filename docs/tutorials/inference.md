<!--
 Copyright 2023–2026 Google LLC

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

(inference)=

# Inference on MaxText

We support inference of MaxText models on vLLM via an [out-of-tree](https://github.com/vllm-project/tpu-inference/blob/main/docs/getting_started/out-of-tree.md) model plugin for vLLM. In this guide we will show how to leverage this for offline inference, online inference and for use in our reinforcement learning (RL) workflows.

> **_NOTE:_**
> The commands in this tutorial assume access to a v6e-8 VM.

# Installation

Follow the instructions in [install maxtext](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) to install MaxText. For this inference tutorial on TPU (which uses vLLM), you must install `maxtext[tpu-post-train]`, as it includes the required adapter plugin. We recommend installing from PyPI to ensure you have the latest stable version of dependencies.

After finishing the installation, ensure that the MaxText on vLLM adapter plugin has been installed. To do so, run the following command:

```bash
uv pip show maxtext_vllm_adapter
```

You should see an output similar to the following if everything has been installed correctly:

```bash
Using Python 3.12.12 environment at: maxtext_venv
Name: maxtext-vllm-adapter
Version: 0.1.0
Location: ~/maxtext/maxtext_venv/lib/python3.12/site-packages
Requires:
Required-by:
```

If the plugin is not installed, please run the install post training extra dependencies script again with the following command:

```bash
install_tpu_post_train_extra_deps
```

# Offline Inference

We include a script for convenient offline inference of MaxText models in `src/maxtext/inference/vllm_decode.py`. This is helpful to ensure correctness of MaxText checkpoints. This script invokes the [`LLM`](https://docs.vllm.ai/en/latest/serving/offline_inference/#offline-inference) API from vLLM.

> **_NOTE:_**
> You will need to convert a checkpoint from HuggingFace in order to run the command. Do so first by following the steps in the [convert checkpoint](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/convert_checkpoint.html) tutorial.

> **_NOTE:_**
> The remainder of this tutorial assumes that the path to the converted MaxText checkpoint is stored in \$CHECKPOINT_PATH.

An example of how to run this script can be found below:

```bash
  python3 -m maxtext.inference.vllm_decode \
      model_name=qwen3-30b-a3b \
      tokenizer_path=Qwen/Qwen3-30B-A3B \
      load_parameters_path=$CHECKPOINT_PATH \
      vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
      ici_tensor_parallelism=8 \
      enable_dp_attention=True \
      hbm_utilization_vllm=0.5 \
      prompt="Suggest some famous landmarks in London." \
      decode_sampling_temperature=0.0 \
      decode_sampling_nucleus_p=1.0 \
      decode_sampling_top_k=0.0 \
      use_chat_template=True
```

In the command above we pass in the `vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}'` argument. This argument tells vLLM to use the MaxText implementation of the target model architecture.

# Online Inference

We can also run online inference (an inference server) running a MaxText model by using the [`vllm serve`](https://docs.vllm.ai/en/stable/cli/serve/) API. In order to invoke this with a MaxText model, we provide the following additional arguments:

```bash
# --hf-overrides specifies that the MaxText model architecture should be used.
--hf-overrides "{\"architectures\": [\"MaxTextForCausalLM\"]}"

# --additional-config passes in "maxtext_config" which contains overrides to initialize the model.
--additional-config "{\"maxtext_config\": {\"model_name\": \"qwen3-235b-a22b\", \"log_config\": false, \"load_parameters_path\": \"$CHECKPOINT_PATH\"}"
```

An example of how to run `vllm serve` can be found below:

```bash
vllm serve Qwen/Qwen3-30B-A3B \ 
  --seed 42 \
  --max-model-len=5120 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --max-num-batched-tokens 4096 \
  --max_num_seqs 128 \
  --hf-overrides "{\"architectures\" [\"MaxTextForCausalLM\"]}" \
  --additional-config "{\"maxtext_config\": {\"model_name\": \"qwen3-30b-a3b\", \"log_config\": false, \"load_parameters_path\": \"$CHECKPOINT_PATH\}}"
```

In a separate bash shell, you can send a request to this server by running the following:

```bash
curl http://localhost:8000/v1/completions \
   -H "Content-Type: application/json" \
   -d '{
       "model": "Qwen/Qwen3-30B-A3B",
       "prompt": ["Suggest some famous landmarks in London."],
       "max_tokens": 4096,
       "temperature": 0
   }'
```

# Reinforcement Learning (RL)

> **_NOTE:_**
> Please refer to the [reinforcement learning tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html) to get started with reinforcement learning on MaxText.

> **_NOTE:_**
> You will need a HuggingFace token to run this command in addition to a MaxText model checkpoint. Please see the following [guide](https://huggingface.co/docs/hub/en/security-tokens) to generate one.

To use a MaxText model architecture for samplers in reinforcement learning algorithms like GRPO, we can override the vLLM model architecture and pass in MaxText specific config arguments similar to the [online inference](https://maxtext.readthedocs.io/en/latest/tutorials/inference.html#online-inference) use-case. An example of an RL command using the MaxText model for samplers can be found below:

```bash
python3 -m src.maxtext.trainers.post_train.rl.train_rl \
  model_name=qwen3-0.6b \
  tokenizer_path=Qwen/Qwen3-0.6B \
  run_name=$WORKLOAD \
  base_output_directory=$OUTPUT_PATH \
  hf_access_token=$HF_TOKEN \
  batch_size=4 \
  num_batches=5 \
  scan_layers=True \
  hbm_utilization_vllm=0.4 \
  rollout_data_parallelism=2 \
  rollout_tensor_parallelism=4 \
  allow_split_physical_axes=true \
  load_parameters_path=$CHECKPOINT_PATH \
  vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
  vllm_additional_config='{"maxtext_config": {"model_name": "qwen3-0.6b", "log_config": "false"}}'
```
