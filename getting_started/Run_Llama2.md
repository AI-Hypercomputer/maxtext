<!--
 Copyright 2023 Google LLC

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

## About Llama2

MaxText supports [Llama2](https://llama.meta.com/llama2) pretraining, finetuning and decoding for its 7B and 70B flavors. To get started on decoding and finetuning of Llama2, you will first need to download weights along with its tokenizer from [Meta](https://llama.meta.com/llama-downloads). 

The file [test_llama2_7b.sh](https://github.com/google/maxtext/blob/main/end_to_end/test_llama2_7b.sh) provides details on how to convert the PyTorch weights in orbax checkpoint format, and thereafter use it for running decoding and finetuning. [test_llama2_7b.sh](https://github.com/google/maxtext/blob/main/end_to_end/test_llama2_7b.sh) also shows how to run pretraining and also how to run decoding on the finetuned model checkpoint. 

### MaxText supports pretraining and finetuning with high performance.

Model Flop utilization for training on v5e and v5p and v4 TPUs with MaxText.


| Model      | v4-128 (bf16)  | v5p-128 (bf16) | v5e-256 (bf16) |
| ---------- | -------------- | -------------- | -------------- |
| Llama2-70b | 57%            | 65%            | 57%         |
