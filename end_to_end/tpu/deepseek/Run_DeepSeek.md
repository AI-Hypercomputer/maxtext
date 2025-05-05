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

# DeepSeek

DeepSeek is a novel family of open-weights sparse MoE models by DeepSeek AI. DeepSeek-V3 features advanced techniques, including Multi-Head Latent Attention (MLA), finer-grained and shared experts, Multi-Token Prediction (MTP), and FP8 mixed precision designed for enhanced efficiency and performance. The currently supported models are DeepSeek V3 (671B) and DeepSeek V2-Lite (16B).

Please note:
* MTP and FP8 mixed precision is not supported yet.
* To leverage MLA with Flash Attention, ensure you have the latest JAX version.
* The provided TPU configurations are examples and not mandatory.


## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining with V3 on v5p-256.

```sh
python3 -m MaxText.train MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=matmul_pre_training \
    per_device_batch_size=4 \
    enable_checkpointing=false \
    model_name=deepseek3-671b \
    ici_fsdp_parallelism=128 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    dataset_type=synthetic
```


## Checkpoint conversion
To get started, follow the instructions at HuggingFace ([V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)) to download the model. Currently, for V3, please convert it from FP8 to BF16 using script [here](https://github.com/deepseek-ai/DeepSeek-V3/blob/a878eada08ea6913f5a2ae80a43afeffdef082ef/inference/fp8_cast_bf16.py). Once downloaded and converted to BF16:
* run [convert_deepseek_ckpt.py](../../../MaxText/convert_deepseek_ckpt.py) to convert the checkpoint for MaxText compatibility in [Orbax](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html) for training and fine-tuning.
* run [convert_deepseek_unscanned_ckpt.py](../../../MaxText/convert_deepseek_unscanned_ckpt.py) to convert the checkpoint to unscanned version in Orbax for decoding.


## Fine-tuning

After you have a MaxText compatible checkpoint, you could fine-tune it with different datasets. 

One example command to run general finetuning with V3 on v5p-256.

```sh
python3 -m MaxText.train MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    dataset_path=${DATASET_PATH} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=matmul_fine_tuning \
    per_device_batch_size=4 \
    model_name=deepseek3-671b \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    enable_checkpointing=true \
    ici_expert_parallelism=128 \
    ici_fsdp_parallelism=1
```

One example command to run supervised finetuning with V3 on v5p-256. Supervised fine-tuning is only working with HuggingFace conversational datasets. And, you can customize the dataset path using the `hf_path` config and provide your access token with `hf_access_token` config.

```sh
python3 -m MaxText.sft_trainer MaxText/configs/sft.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=matmul_supervised_fine_tuning \
    per_device_batch_size=4 \
    model_name=deepseek3-671b \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    enable_checkpointing=true \
    ici_expert_parallelism=128 \
    ici_fsdp_parallelism=1 \
    dataset_type=hf
```

## Decoding
One example command to run supervised finetuning with V3 on v5p-256 with unscanned checkpoint for fast decoding. When decoding with a supervised fine-tuned checkpoint, format your prompt as `prompt='<user>your text</user> <assistant>'`.

```sh
python3 -m MaxText.decode MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=decode \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    model_name=deepseek3-671b \
    max_prefill_predict_length=100 \
    max_target_length=1024 \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    ici_tensor_parallelism=128 \
    ici_fsdp_parallelism=1 \
    prompt="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and outputs are all vectors. The output is " \
    scan_layers=False
```

## Supported MoE strategy
* Dropless
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.
* Dropping implementation with flag `sparse_matmul=False` and reasonable `capacity_factor`, commonly used from 1 to 1.25.

See more examples in scripts for [V3](v3-671b/test_deepseek.sh) and [V2-Lite](v2-16b/test_deepseek.sh).
