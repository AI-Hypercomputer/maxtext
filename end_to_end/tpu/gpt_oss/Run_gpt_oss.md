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

# GPT-OSS

The [gpt-oss models](https://openai.com/index/introducing-gpt-oss/) are OpenAI's first open-weight language models since GPT‑2, released in August 2025. They leverage mixture-of-experts (MoE) architecture.

<!-- 
is a family of state-of-the-art open-weight language models that deliver strong real-world performance at low cost by OpenAI. DeepSeek-V3 features advanced techniques, including attnetion with sink.  -->


The currently supported models are 

- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)



## Checkpoint conversion

For finetuning, supervised finetuning, or decoding, we need convert the checkpoint from HuggingFace.


1. To get started, follow the instructions at HuggingFace to download the model. For [faster download](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads), you may use `hf_xet` or `hf_transfer`.

```
huggingface-cli download [openai/gpt-oss-20b|openai/gpt-oss-120b] --local-dir <local_mxfp4_path> --token $HF_TOKEN
```


2. Please convert it from MXFP4 to BF16 using script [dequantize_mxfp4.py](../../../src/MaxText/scratch_code/dequantize_mxfp4.py) on gpu.

```
python3 dequantize_mxfp4.py --input-path=<local_mxfp4_path> --output-path=<local_bf16_path> --dtype-str=bf16
```


3. Once downloaded and converted to BF16:
* run [convert_gpt_oss_ckpt.py](../../../src/MaxText/convert_gpt_oss_ckpt.py) to convert the checkpoint for MaxText compatibility in [Orbax](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html) for training and fine-tuning. 

```
python3 -m MaxText.convert_gpt_oss_ckpt --base-model-path <local_bf16_path> \
    --maxtext-model-path <GCS/path/to/scanned/maxtext/ckpt> --model-size [gpt-oss-20b|gpt-oss-120b]
```

* run [convert_gpt_oss_unscanned_ckpt.py](../../../src/MaxText/convert_gpt_oss_unscanned_ckpt.py) to convert the checkpoint to unscanned version in Orbax for decoding.

```
python3 -m MaxText.convert_gpt_oss_unscanned_ckpt --base-model-path <local_bf16_path> \
    --maxtext-model-path <GCS/path/to/unscanned/maxtext/ckpt> --model-size [gpt-oss-20b|gpt-oss-120b]
```


## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining with V3 on v5p-256.

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
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



## Fine-tuning

After you have a MaxText compatible checkpoint, you could fine-tune it with different datasets. 

One example command to run general finetuning with V3 on v5p-256.

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
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
python3 -m MaxText.sft_trainer src/MaxText/configs/sft.yml \
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
One example command to run decoding with V3 on v5p-256 with unscanned checkpoint for fast decoding.

```sh
python3 -m MaxText.decode src/MaxText/configs/base.yml \
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

## Correctness
To verify the correctness of the model implementation, we perform **Logit Comparison**. We compare the logits generated by our implementation against those from a HuggingFace implementation for a set of given prompts.

One example command to generate golden logits from HuggingFace for V2-Lite.

```sh
python3 -m MaxText.scratch_code.generate_hf_golden_logits \
    --model-id=openai/gpt-oss-20b \
    --output-path=golden_gpt-oss-20b.jsonl \
    --prompts='I love to;Today is a;What is the'
```

You should be able to see logs like below:

```
...
File is stored locally at golden_DeepSeek-V2-Lite.jsonl.
```

Run command below to compare logits between HuggingFace and MaxText.

```sh
python3 -m tests.forward_pass_logit_checker \
    src/MaxText/configs/base.yml \
    load_parameters_path=${CONVERTED_CHECKPOINT} \
    run_name=forward_pass_test_deepseek2-16b \
    per_device_batch_size=1 \
    model_name=deepseek2-16b \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    dataset_type=synthetic \
    scan_layers=false \
    sparse_matmul=False \
    dtype=float32 \
    activations_in_float32=true \
    matmul_precision=high \
    --max_kl_div=2e-4 \
    --golden_logits_path=${PWD}/golden_DeepSeek-V2-Lite.jsonl
```


## Supported MoE strategy

* Dropless
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.


Note: Dropping implementation is not supported. You should avoid using flag `sparse_matmul=False` and `capacity_factor != -1`.

See more examples in scripts for [V3](v3-671b/test_deepseek.sh) and [V2-Lite](v2-16b/test_deepseek.sh).