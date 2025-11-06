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

The gpt-oss models are OpenAI's first open-weight language models since GPT‑2, released in August 2025. The models use mixture-of-experts (MoE) transformer architecture. You can find more information in the [blog](https://openai.com/index/introducing-gpt-oss/) and the [model card](https://arxiv.org/abs/2508.10925). The currently supported models are [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) and [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b).



## Checkpoint conversion

For finetuning, supervised finetuning, or decoding, we need convert the checkpoint from HuggingFace.


1. To get started, follow the instructions at HuggingFace to download the model. For faster download, you may use `hf_xet` or `hf_transfer`. The model weights are quantized in MXFP4.
```
hf download [openai/gpt-oss-20b|openai/gpt-oss-120b] --local-dir <local_mxfp4_path>
```


2. Please convert it from MXFP4 to BF16 using script [dequantize_mxfp4.py](../../../src/MaxText/utils/ckpt_scripts/dequantize_mxfp4.py) on gpu.

```
python3 -m MaxText.utils.ckpt_scripts.dequantize_mxfp4 --input-path=<local_mxfp4_path> --output-path=<local_bf16_path> --dtype-str=bf16
```


3. Once downloaded and converted to BF16:
* run [convert_gpt_oss_ckpt.py](../../../src/MaxText/utils/ckpt_scripts/convert_gpt_oss_ckpt.py) to convert the checkpoint for MaxText compatibility in [Orbax](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html) scanned format for training and fine-tuning. 

```
python3 -m MaxText.utils.ckpt_scripts.convert_gpt_oss_ckpt --base-model-path <local_bf16_path> \
    --maxtext-model-path <GCS/path/to/scanned/maxtext/ckpt> --model-size [gpt-oss-20b|gpt-oss-120b]
```

* run [convert_gpt_oss_unscanned_ckpt.py](../../../src/MaxText/utils/ckpt_scripts/convert_gpt_oss_unscanned_ckpt.py) to convert the checkpoint to unscanned format in Orbax for decoding.

```
python3 -m MaxText.utils.ckpt_scripts.convert_gpt_oss_unscanned_ckpt --base-model-path <local_bf16_path> \
    --maxtext-model-path <GCS/path/to/unscanned/maxtext/ckpt> --model-size [gpt-oss-20b|gpt-oss-120b]
```


## Pretraining
You can train from scratch to generate a new checkpoint. One example command to run pretraining with gpt-oss-20b on v5p-8.

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=megablox_pre_training \
    model_name=gpt-oss-20b \
    tokenizer_type=huggingface \
    tokenizer_path=openai/gpt-oss-20b \
    dataset_type=synthetic \
    enable_checkpointing=false \
    attention=flash \
    sparse_matmul=True \
    megablox=True \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=4 \
    steps=5 \
    max_target_length=1024 \
    ici_fsdp_parallelism=4
```

## Finetuning

After you have a MaxText-compatible scanned checkpoint, you could finetune it with different datasets. 

One example command to run general finetuning with gpt-oss-20b on v5p-8.

```sh
python3 -m MaxText.train src/MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=megablox_fine_tuning \
    model_name=gpt-oss-20b \
    tokenizer_type=huggingface \
    tokenizer_path=openai/gpt-oss-20b \
    dataset_path=${DATASET_PATH} \
    enable_checkpointing=true \
    async_checkpointing=false \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    scan_layers=True \
    attention=flash \
    sparse_matmul=True \
    megablox=True \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=4 \
    steps=5 \
    max_target_length=1024 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4
```

One example command to run supervised finetuning with gpt-oss-20b on v5p-8. Supervised finetuning is only working with HuggingFace conversational datasets. And, you can customize the dataset path using the `hf_path` config. If using [gated dataset](https://huggingface.co/docs/hub/en/datasets-gated) or [gated model](https://huggingface.co/docs/hub/en/models-gated), you need additionally provide the access token with `hf_access_token` config.

```sh
python3 -m MaxText.sft_trainer src/MaxText/configs/sft.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=megablox_supervised_fine_tuning \
    model_name=gpt-oss-20b \
    tokenizer_type=huggingface \
    tokenizer_path=openai/gpt-oss-20b \
    dataset_type=hf \
    enable_checkpointing=true \
    async_checkpointing=false \
    load_parameters_path=${SCANNED_CKPT_PATH} \
    scan_layers=True \
    attention=flash \
    sparse_matmul=True \
    megablox=True \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=4 \
    steps=5 \
    max_target_length=1024 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4
```

## Decoding
One example command to run decoding with gpt-oss-20b on v5p-8 with unscanned checkpoint for fast decoding.

```sh
python3 -m MaxText.decode src/MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=decode \
    model_name=gpt-oss-20b \
    tokenizer_type=huggingface \
    tokenizer_path=openai/gpt-oss-20b \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=False \
    attention=dot_product \
    sparse_matmul=True \
    megablox=True \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    per_device_batch_size=1 \
    max_prefill_predict_length=64 \
    max_target_length=128 \
    prompt="I love to" \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=4
```

## Correctness
To verify the correctness of the model implementation, we perform Logit Comparison. We compare the logits generated by our implementation against those from a HuggingFace implementation for a set of given prompts.

One example command to generate golden logits from HuggingFace for gpt-oss-20b:

```sh
python3 -m MaxText.scratch_code.generate_hf_golden_logits \
    --model-id=openai/gpt-oss-20b \
    --output-path=golden_data_gpt-oss-20b.jsonl \
    --prompts='I love to;Today is a;What is the' \
    --hf-model-path=<local_bf16_path>
```
The hf model path should contains the HuggingFace checkpoint in bf16, dequantized from the previous step. You should also add the [config.json](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) and remove "quantization_config".

You should be able to see logs like below:
```
File is stored locally at golden_data_gpt-oss-20b.jsonl.
```

Run command below to compare logits between HuggingFace and MaxText.

```sh
python3 -m tests.forward_pass_logit_checker \
    src/MaxText/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=forward_logits_check \
    model_name=gpt-oss-20b \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=false \
    attention=dot_product \
    sparse_matmul=True \
    megablox=True \
    per_device_batch_size=1 \
    max_target_length=4 \
    max_prefill_predict_length=4 \
    dtype=float32 \
    --atol=0.1 --rtol=0.1 \
    --max_kl_div=3e-4 \
    --golden_logits_path=${PWD}/golden_data_gpt-oss-20b.jsonl
```
This is to check unscanned checkpoint. To check scanned checkpoint, you can use `load_parameters_path=${SCANNED_CKPT_PATH} scan_layers=true`.


## Supported MoE strategy

* Dropless
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.


Note: Dropping implementation is not supported. You should avoid using flag `sparse_matmul=False` and `capacity_factor!=-1`.

See more examples in scripts for [20b](20b/test_gpt_oss.sh) and [120b](120b/test_gpt_oss.sh).