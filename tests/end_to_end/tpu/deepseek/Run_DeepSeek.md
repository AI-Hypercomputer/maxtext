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

# DeepSeek

DeepSeek is a novel family of open-weights sparse MoE models by DeepSeek AI. The currently supported models are DeepSeek V2-Lite (16B), DeepSeek V3 (671B), DeepSeek R1 (671B), DeepSeek V3.1 (671B), DeepSeek V3.2 (671B), and DeepSeek V4-Flash (284B).

* DeepSeek-V3 features advanced techniques, including Multi-Head Latent Attention (MLA), finer-grained and shared experts, Multi-Token Prediction (MTP), and FP8 mixed precision designed for enhanced efficiency and performance.

* DeepSeek R1 also uses V3 architecture. It utilizes cold-start data and large-scale reinforcement learning to incentivize chain-of-thought reasoning without relying solely on supervised fine-tuning.

* DeepSeek-V3.1 shares the same architecture as V3, but features an improved checkpoint that supports hybrid thinking modes, improved performance in agentic tasks, and higher thinking efficiency.

* DeepSeek-V3.2 introduces [DeepSeek Sparse Attention](https://arxiv.org/pdf/2512.02556) (DSA), successfully reduces computational complexity while preserving model performance in long-context scenarios.

* DeepSeek V4-Flash introduces radical architectural shifts, moving to Hybrid Attention (combining Compressed Sparse Attention and Heavily Compressed Attention) and Manifold-Constrained Hyper-Connections (mHC).

**Please note:**
* To leverage MLA with Flash Attention, ensure you have the latest JAX version.
* The provided TPU configurations are examples and not mandatory.
* For V3.1 & R1, use existing V3 671B model configurations, as it shares the same architecture.

## Checkpoint conversion
To get started, follow the instructions at HuggingFace ([V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)) to download the model. Currently for V3, V3.1, and R1, it uses mixed precision fp8 & bf16 weights. To convert all FP8 weights to BF16, use the script [here](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/ckpt_scripts/deepseek_dequantize.py). Once downloaded and converted to BF16:
* run [convert_deepseek_family_ckpt.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_ckpt.py) to convert the checkpoint for MaxText compatibility in [Orbax](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html) for training and fine-tuning. When converting a checkpoint with MTP layers (like DeepSeek-V3), be sure to add the `--enable_mtp` flag to process them correctly.
* run [convert_deepseek_family_unscanned_ckpt.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_unscanned_ckpt.py) to convert the checkpoint to unscanned version in Orbax for decoding.

### Checkpoint conversion for V3.2 and V4
For V3.2 (671B) and V4-Flash (284B), we use the unified conversion script `src/maxtext/checkpoint_conversion/to_maxtext.py`.

#### 1. Download Model Weights
Download the Hugging Face weights to your local environment:
* For V3.2: [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
* For V4-Flash: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

#### 2. Dequantize Weights (if in FP8/FP4)
If downloading the quantized weights, convert them from FP8/FP4 to BF16 using [deepseek_dequantize.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_dequantize.py) on CPU:  

```bash
python3 -m maxtext.checkpoint_conversion.standalone_scripts.deepseek_dequantize \
    --input-fp8-hf-path=<local_fp8_path> \
    --output-bf16-hf-path=<local_bf16_path>  
```
*(Alternatively, use the official DeepSeek script [fp8_cast_bf16.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) to convert on GPU).*

#### 3. Convert to MaxText-compatible Orbax format
Execute the following command to finalize the conversion. Ensure your environment variables (`$BASE_OUTPUT_PATH`, `$HF_TOKEN`, and `$BF16_LOCAL_PATH`) are exported before running. Set `MODEL_NAME` to `deepseek3.2-671b` or `deepseek4-284b`. Setting `scan_layers=true` generates scanned Orbax format for training and fine-tuning. Setting `scan_layers=false` generates unscanned format in Orbax for decoding.

> [!TIP]
> For large models like V3.2 and V4, you can optionally speed up the conversion drastically on large CPU machines by sharding checkpoint across simulated cpu devices and eagerly loading tensors. E.g., appending `--simulated_cpu_devices_count=16 --lazy_load_tensors=False` to the command.


```bash
# scanned
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    scan_layers=true \
    base_output_directory=$BASE_OUTPUT_PATH/scanned \
    hf_access_token=$HF_TOKEN \
    hardware=cpu \
    skip_jax_distributed_system=True \
    --hf_model_path=$BF16_LOCAL_PATH \
    --save_dtype=bfloat16

# unscanned
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    scan_layers=false \
    attention=dot_product \
    base_output_directory=$BASE_OUTPUT_PATH/unscanned \
    hf_access_token=$HF_TOKEN \
    hardware=cpu \
    skip_jax_distributed_system=True \
    --hf_model_path=$BF16_LOCAL_PATH \
    --save_dtype=bfloat16
```


## Pre-training
You can train from scratch or resume from a converted scanned checkpoint. The pre-training process and options are similar across all DeepSeek models.

Here is an example command template. Export the variables `MODEL_NAME` (e.g., `deepseek3-671b`, `deepseek4-284b`), `TOKENIZER_PATH` (e.g., `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V4-Flash`), and customize other arguments such as `attention` (`flash` or `dot_product`), `opt_type` (`sgd` or `adamw`), and parallelism topology configs to match your model and setup:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=matmul_pre_training \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    model_name=${MODEL_NAME} \
    ici_fsdp_parallelism=-1 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=${TOKENIZER_PATH} \
    attention=flash \ # DSV4 does not yet support flash attention, need to use dot_product for now.
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
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=matmul_fine_tuning \
    dataset_path=${DATASET_PATH?} \
    load_parameters_path=${SCANNED_CKPT_PATH?} \
    per_device_batch_size=1 \
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

Fine-tuning with MTP on v5p-256

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=deepseek_mtp_finetuning \
    dataset_path=${DATASET_PATH?} \
    load_parameters_path=${SCANNED_CKPT_PATH?} \
    per_device_batch_size=1 \
    model_name=deepseek3-671b \
    steps=10000 \
    max_target_length=2048 \
    ici_fsdp_parallelism=128 \
    attention=flash \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    # MTP-specific flags
    mtp_num_layers=1 \
    mtp_loss_scaling_factor=0.1
```

One example command to run supervised finetuning with V3 on v5p-256. Supervised fine-tuning is only working with HuggingFace conversational datasets. And, you can customize the dataset path using the `hf_path` config and provide your access token with `hf_access_token` config.

```sh
python3 -m maxtext.trainers.post_train.sft.train_sft_native src/maxtext/configs/post_train/sft.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    load_parameters_path=${SCANNED_CKPT_PATH?} \
    run_name=matmul_supervised_fine_tuning \
    per_device_batch_size=1 \
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

## Continued pre-training for V3.2 Sparse Attention

**DeepSeek Sparse Attention (DSA)** enhances the Multi-Head Latent Attention (MLA) architecture by introducing a **Lightning Indexer**, which selects the top-k tokens for attention. Note that Indexer is activated only if `max_target_length` > `indexer_topk` (2048).

DeepSeek-V3.2 is instantiated from DeepSeek-V3.1 and undergoes continued pre-training to adapt this indexer via a two-stage strategy: **Dense Warm-up** and **Sparse Training**.

1. **Dense Warmup Stage**
The indexer is trained exclusively using dense indexer loss while all other model parameters remain frozen.  
```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=indexer_dense_warmup \
    model_name=deepseek3.2-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3.2 \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    async_checkpointing=false \
    ici_fsdp_parallelism=128 \
    steps=5 \
    # Indexer is activated only if max_target_length > indexer_topk (2048)
    max_target_length=4096 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_type=synthetic \
    # Indexer training specific flags
    indexer_loss_scaling_factor=0.01 \
    indexer_sparse_training=False \
    trainable_parameters_mask=['.*indexer.*']
```
2. **Sparse Training Stage**
The indexer is trained with sparse indexer loss, while the remaining model parameters are unfrozen and updated using standard language modeling loss.  
```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=indexer_sparse_training \
    model_name=deepseek3.2-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3.2 \
    per_device_batch_size=1 \
    enable_checkpointing=false \
    async_checkpointing=false \
    ici_fsdp_parallelism=128 \
    steps=5 \
    # Indexer is activated only if max_target_length > indexer_topk (2048)
    max_target_length=4096 \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_type=synthetic \
    # Indexer training specific flags
    indexer_loss_scaling_factor=0.01 \  
    indexer_sparse_training=True
```

## Decoding
One example command to run decoding with V3 on v5p-256 with unscanned checkpoint for fast decoding.

```sh
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=decode \
    model_name=deepseek3-671b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V3 \
    hf_access_token=${HF_TOKEN?} \
    load_parameters_path=${UNSCANNED_CKPT_PATH?} \
    scan_layers=False \
    enable_checkpointing=true \
    async_checkpointing=false \
    per_device_batch_size=1 \
    max_prefill_predict_length=100 \
    max_target_length=1024 \
    attention=dot_product \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=False \
    sparse_matmul=False \
    ici_tensor_parallelism=128 \
    ici_fsdp_parallelism=1 \
    prompt="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and outputs are all vectors. The output is "
```

## Correctness
To verify the correctness of the model implementation, we perform two primary checks:

* **Logit Comparison**: We compare the logits generated by our implementation against those from a HuggingFace implementation for a set of given prompts.
* **MMLU Score Validation**: We validate the MMLU score against established benchmarks.

One example command to generate golden logits from HuggingFace for V2-Lite.

```sh
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
    --model-id=deepseek-ai/DeepSeek-V2-Lite \
    --output-path=golden_DeepSeek-V2-Lite.jsonl \
    --prompts='I love to;Today is a;What is the'
```

You should be able to see logs like below:

```
...
File is stored locally at golden_DeepSeek-V2-Lite.jsonl.
```

Run command below to compare logits between HuggingFace and MaxText.

```sh
python3 -m tests.utils.forward_pass_logit_checker \
    src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=forward_pass_test \
    model_name=deepseek2-16b \
    tokenizer_type=huggingface \
    tokenizer_path=deepseek-ai/DeepSeek-V2-Lite \
    load_parameters_path=${UNSCANNED_CKPT_PATH?} \
    scan_layers=false \
    per_device_batch_size=1 \
    max_prefill_predict_length=4 \
    max_target_length=4 \
    sparse_matmul=False \
    dtype=float32 \
    activations_in_float32=true \
    matmul_precision=high \
    --max_kl_div=2e-4 \
    --golden_logits_path=golden_DeepSeek-V2-Lite.jsonl
```

To run MMLU benchmarks and validate the model's performance, follow the instructions provided [here](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/api_server/README.md).

### Logit comparison for V4-284b

To verify correctness for the DeepSeek V4 model, we compare the logits against the HuggingFace implementation.

Run the forward pass logit checker command below. The test uses `indexer_topk=4` to cover all pathways in the model without excessive sequence lengths in testing. To cover each pathway in the model, we need to set the prompt length such that: sliding window (`prompt_len > sliding_window_size`), HCA (`prompt_len / hca_compression_factor > 1`), and CSA (`prompt_len / csa_compression_factor > indexer_topk`).

**Important Note for Golden Logit Generation**: To generate golden logits with `indexer_topk=4`, you must modify `config.json` in your local HuggingFace checkpoint directory, changing `"index_topk": 512` to `"index_topk": 4` before running the generation script.

One example command to generate golden logits from HuggingFace for DeepSeek V4-Flash (using a prompt length of 512 tokens to trigger sliding window, CSA, and HCA layer requirements):

```sh
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
    --model-id=deepseek-ai/DeepSeek-V4-Flash \
    --output-path=golden_data_deepseek4-284b_proper_512.jsonl \
    --prompts='<Insert prompt here>' \
    --hf-model-path=$BF16_LOCAL_PATH
```

```sh
python3 -m tests.utils.forward_pass_logit_checker \
    src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=ds4-parity \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    scan_layers=False \
    attention=dot_product \
    per_device_batch_size=1 \
    model_name=deepseek4-284b \
    max_target_length=1024 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=-1 \
    weight_dtype=bfloat16 \
    dtype=bfloat16 \
    activations_in_float32=false \
    matmul_precision=highest \
    float32_logits=false \
    float32_qk_product=false \
    override_model_config=True \
    indexer_topk=4 \
    enable_nnx=false \
    pure_nnx=false \
    pure_nnx_decoder=false \
    --golden_logits_path=golden_data_deepseek4-284b_proper_512.jsonl \
    --max_kl_div=0.75
```

## Supported MoE strategy
* Dropless
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.
* Dropping implementation with flag `sparse_matmul=False` and reasonable `capacity_factor`, commonly used from 1 to 1.25.

See more examples in scripts for [V2-Lite](v2-16b), [V3](v3-671b), and [V3.2](v3.2-671b).
