<!--
 # Copyright 2023–2025 Google LLC
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

# Llama4

​Meta's Llama 4 is the latest generation of its open-source large language models (LLMs), unveiled in April 2025. These models are designed to be natively multimodal and multilingual, incorporating a mixture-of-experts (MoE) architecture to enhance performance and efficiency.  The currently supported models are:
* LLama4 Scout (17B-16E)
* Llama4 Scout (17B-16E-Instruct)
* LLama4 Maverick (17B-128E)
* Llama4 Maverick (17B-128E-Instruct)


## Checkpoint conversion
Currently, we support converting both [PyTorch](https://www.llama.com/) and [HuggingFace](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164) checkpoints.  Note that we recommend using the `huggingface-cli download` command with environment variable
`HF_HUB_ENABLE_HF_TRANSFER=1` to download the models.  You can also find a copy of the PyTorch model weights in the HuggingFace repo by looking
for the models with the `-Original` suffix ([example](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct-Original)).

Once you have downloaded the models, you can run the following command to generate an unscanned checkpoint (preferred for decoding):

```
JAX_PLATFORMS=CPU python -m MaxText.llama4_ckpt_unscanned --base-model-path [PATH_TO_CHECKPOINT_DIR] --maxtext-model-path [DESIRED_MAXTEXT_CHECKPOINT_OUTPUT_DIR]  --model-size llama4-17b-16e [--huggingface-checkpoint]
```

Or the following command to generate a scanned checkpoint (preferred for training):
```
JAX_PLATFORMS=CPU python -m MaxText.llama_or_mistral_ckpt --base-model-path [PATH_TO_CHECKPOINT_DIR] --maxtext-model-path [DESIRED_MAXTEXT_CHECKPOINT_OUTPUT_DIR]  --model-size llama4-17b-16e
```

## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining with Llama4 Maverick on a v5p-128.

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=matmul_pre_training \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    model_name=llama4-17b-128e \
    ici_fsdp_parallelism=-1 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    dataset_type=synthetic
```


## Decoding
In order to run an example decoding, you can use a command such as the following:

```
python3 -m MaxText.decode src/MaxText/configs/base.yml scan_layers=false base_output_directory=... load_parameters_path=... run_name=... model_name=llama4-17b-16e force_unroll=false weight_dtype=bfloat16 sparse_matmul=false megablox=false tokenizer_path="meta-llama/Llama-4-Scout-17B-16E"  max_target_length=16 max_prefill_predict_length=4 per_device_batch_size=2 prompt="I love to" attention=dot_product
```

## Supported MoE strategy
* Dropless
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.