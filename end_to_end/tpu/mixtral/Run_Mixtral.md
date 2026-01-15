<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# Mixtral

[Mixtral](https://mistral.ai/news/mixtral-of-experts/) is a state-of-the-art AI model developed by Mistral AI, utilizing a sparse mixture-of-experts (MoE) architecture.


To get started, follow the instructions at [mistral-inference](https://github.com/mistralai/mistral-inference) to download the model. Once downloaded, run [llama_or_mistral_ckpt.py](../../../src/MaxText/llama_or_mistral_ckpt.py) to convert the checkpoint for MaxText compatibility. You can then proceed with decoding, pretraining, and finetuning. You could find Mixtral 8x7B example in the [end_to_end/tpu/mixtral/8x7b](../mixtral/8x7b) test scripts.


Additionally, Mixtral integrates with [MegaBlocks](https://arxiv.org/abs/2211.15841), an efficient dropless MoE strategy, which can be activated by setting both sparse_matmul and megablox flags to True (default).


## MaxText supports pretraining and finetuning with high performance

Model Flop utilization for training on v5p TPUs.

| Model size    | Accelerator type | TFLOP/chip/sec | Model flops utilization (MFU) |
| ------------ | -------------- | --------------  | -------------- |
| Mixtral 8X7B | v5p-128       | 251.94 | 54.89% |


