<!--
 Copyright 2025 Google LLC

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

# Llama4

​Meta's Llama 4 is the latest generation of its open-source large language models (LLMs), unveiled in April 2025. These models are designed to be natively multimodal and multilingual, incorporating a mixture-of-experts (MoE) architecture to enhance performance and efficiency.  The currently supported models are:
* LLama4 Scout (17B-16E)
* Llama4 Scout (17B-16E-Instruct)


## Checkpoint conversion
Currently, we support converting PyTorch (`.pth`) checkpoints, which can be downloaded from Meta [here](https://www.llama.com/).

Once you have downloaded the Scout models, you can run the following command to generate an unscanned checkpoint (preferred for decoding):

```
JAX_PLATFORMS=CPU python -m MaxText.llama4_ckpt_unscanned --base-model-path [PATH_TO_CHECKPOINT_DIR] --maxtext-model-path [DESIRED_MAXTEXT_CHECKPOINT_OUTPUT_DIR]  --model-size llama4-17b-16e
```

Or the following command to generate a scanned checkpoint (preferred for training):
```
JAX_PLATFORMS=CPU python -m MaxText.llama_or_mistral_ckpt --base-model-path [PATH_TO_CHECKPOINT_DIR] --maxtext-model-path [DESIRED_MAXTEXT_CHECKPOINT_OUTPUT_DIR]  --model-size llama4-17b-16e
```



## Decoding
In order to run an example decoding, you can use a command such as the following:

```
python -m MaxText.decode MaxText/configs/base.yml scan_layers=false base_output_directory=... load_parameters_path=... run_name=... model_name=llama4-17b-16e force_unroll=false weight_dtype=bfloat16 sparse_matmul=false megablox=false tokenizer_path="meta-llama/Llama-4-Scout-17B-16E"  max_target_length=16 max_prefill_predict_length=4 per_device_batch_size=2 prompt="I love to" attention=dot_product
```

## Supported MoE strategy
* Dropless
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.
  * Support for `megablox` and `ragged_dot` is coming soon!