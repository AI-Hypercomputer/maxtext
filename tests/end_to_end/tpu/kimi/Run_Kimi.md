<!--
 # Copyright 2023-2026 Google LLC
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

# Kimi

Kimi is a family of high-performance, open-weights sparse MoE models by Moonshot AI designed for agentic intelligence. Supported models for checkpoint conversion are **Kimi K2 (1T)**, **Kimi K2-Thinking**, **Kimi K2.5** (text branch), and **Kimi K2.6** (text branch). All variants share the same DeepSeek-V3-style architecture (61 layers, 384 routed experts, MLA).

* **[Kimi K2](https://arxiv.org/pdf/2507.20534)** features a massive 1.04 trillion total parameters with 32 billion activated parameters. The architecture is similar to DeepSeek-V3. It utilizes **Multi-Head Latent Attention (MLA)** and an ultra-sparse MoE with **384 experts**, optimized for long-context and agentic tasks.
* **MuonClip Optimizer**: Kimi K2 was trained using the token-efficient **[Muon optimizer](https://kellerjordan.github.io/posts/muon)** combined with a novel **QK-clip** technique to ensure training stability and eliminate loss spikes during large-scale pre-training.
* **Agentic Excellence**: K2 is specifically post-trained using a large-scale agentic data synthesis pipeline and Reinforcement Learning (RL), achieving state-of-the-art performance on benchmarks like Tau2-Bench and SWE-Bench.

## Checkpoint Conversion
1. To get started, download the model from HuggingFace: [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct). Weights are provided in FP8.

```sh
hf download moonshotai/Kimi-K2-Instruct --local-dir $LOCAL_FP8_PATH
```

2. Convert the weights from FP8 to BF16 using script [deepseek_fp8_to_bf16.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_fp8_to_bf16.py) on CPU
```sh
python3 -m maxtext.checkpoint_conversion.standalone_scripts.deepseek_fp8_to_bf16 --input-fp8-hf-path=$LOCAL_FP8_PATH --output-bf16-hf-path=$LOCAL_BF16_PATH
```
Alternatively, we can use the official DeepSeek script [fp8_cast_bf16.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) to convert on GPU.

3. To convert the checkpoint for MaxText compatibility in [Orbax](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html) 
- Run [convert_deepseek_family_ckpt.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_ckpt.py) to convert the checkpoint to scanned format in Orbax for training and fine-tuning.
```sh
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt --model_size kimi-k2-1t --base_model_path $LOCAL_BF16_PATH --maxtext_model_path $GCS_PATH_TO_SAVE
```
- Run [convert_deepseek_family_unscanned_ckpt.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_unscanned_ckpt.py) to convert the checkpoint to unscanned format in Orbax for decoding.
```sh
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_unscanned_ckpt --model_size kimi-k2-1t --base_model_path $LOCAL_BF16_PATH --maxtext_model_path $GCS_PATH_TO_SAVE
```

### Quantized variants: K2-Thinking, K2.5, K2.6

K2-Thinking/K2.5/K2.6 ship routed-expert weights as int4 (compressed-tensors pack-quantized, group_size=32 symmetric); the converter dequantizes them inline to bf16. The FP8→bf16 path above does not apply. Other tensors (attention, shared experts, dense MLP, `lm_head`) are already bf16 in the checkpoint and pass through unchanged.

K2.5 and K2.6 are multimodal wrappers (`KimiK25ForConditionalGeneration`); the converter strips the `language_model.` prefix and silently drops `vision_tower.*` / `mm_projector.*` keys to produce a text-only MaxText checkpoint.

1. Download the model from HuggingFace — pick one:
```sh
hf download moonshotai/Kimi-K2-Thinking --local-dir $LOCAL_HF_PATH   # --model_size kimi-k2-thinking
hf download moonshotai/Kimi-K2.5         --local-dir $LOCAL_HF_PATH   # --model_size kimi-k2.5-text
hf download moonshotai/Kimi-K2.6         --local-dir $LOCAL_HF_PATH   # --model_size kimi-k2.6-text
```

2. Convert directly to Orbax (no intermediate FP8→BF16 pass):
```sh
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt \
    --model_size <kimi-k2-thinking | kimi-k2.5-text | kimi-k2.6-text> \
    --base_model_path $LOCAL_HF_PATH \
    --maxtext_model_path $GCS_PATH_TO_SAVE
```
Use `convert_deepseek_family_unscanned_ckpt.py` with the same `--model_size` for the unscanned (decoding) layout.

> **Note:** The Pre-training / fine-tuning / decoding flows below use `model_name=kimi-k2-1t`. K2-Thinking / K2.5 / K2.6 text branches share K2's architecture, so the same config works — just point `tokenizer_path` at the variant-specific HF tokenizer (e.g. `moonshotai/Kimi-K2.5`) and `load_parameters_path` at the converted checkpoint.

## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pre-training with Kimi K2 on tpu7x-512 with 256 chips. To use **MuonClip optimizer**, you need `optax>=0.2.7` and `tokamax>=0.0.11`.

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=kimi_k2_pre_training \
    per_device_batch_size=16 \
    enable_checkpointing=false \
    model_name=kimi-k2-1t \
    ici_fsdp_parallelism=64 \
    ici_expert_parallelism=8 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=moonshotai/Kimi-K2-Instruct \
    attention=flash \
    use_tokamax_splash=True \
    use_tokamax_gmm=False \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_type=synthetic \
    scan_layers=True \
    use_ring_of_experts=True \
    # muon optimizer
    opt_type=muon \
    muon_consistent_rms=0.2 \
    muon_weight_decay=0.1 \
    # qk clip
    use_qk_clip=True \
    qk_clip_threshold=100
```

## Fine-tuning
After you have a MaxText compatible checkpoint, you can fine-tune Kimi K2. The Kimi team recommends using the **Muon optimizer** during fine-tuning, as it produces the best performance with a Muon-pre-trained checkpoint.

Example command for General Fine-tuning on tpu7x-512:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=kimi_k2_fine_tuning \
    per_device_batch_size=16 \
    enable_checkpointing=true \
    model_name=kimi-k2-1t \
    ici_fsdp_parallelism=64 \
    ici_expert_parallelism=8 \
    steps=5 \
    max_target_length=1024 \
    async_checkpointing=false \
    tokenizer_type=huggingface \
    tokenizer_path=moonshotai/Kimi-K2-Instruct \
    attention=flash \
    use_tokamax_splash=True \
    use_tokamax_gmm=False \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    dataset_path=${DATASET_PATH?} \
    scan_layers=True \
    load_parameters_path=${SCANNED_CHECKPOINT?} \
    use_ring_of_experts=True \
    # muon optimizer
    opt_type=muon \
    muon_consistent_rms=0.2 \
    muon_weight_decay=0.1 \
    # qk clip
    use_qk_clip=True \
    qk_clip_threshold=100
```

## Decoding
Example command to run decoding with Kimi K2. Given its 1T size, high tensor parallelism is recommended.

```sh
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    run_name=kimi_decode \
    model_name=kimi-k2-1t \
    tokenizer_type=huggingface \
    tokenizer_path=moonshotai/Kimi-K2-Instruct \
    hf_access_token=${HF_TOKEN?} \
    load_parameters_path=${UNSCANNED_CKPT_PATH?} \
    scan_layers=False \
    enable_checkpointing=true \
    async_checkpointing=false \
    per_device_batch_size=1 \
    max_target_length=2048 \
    attention=dot_product \
    ici_tensor_parallelism=128 \
    ici_fsdp_parallelism=1 \
    prompt="The primary goal of agentic intelligence is to "
```

## Correctness

To verify the correctness of the Kimi K2 implementation, we perform two primary validation steps:

  * **Logit Comparison**: We compare the logits generated by our implementation against those from a HuggingFace implementation for a set of given prompts.
  * **MMLU Score Validation**: We validate the MMLU score against established benchmarks. 

### Logit Comparison

Use the following example to generate "golden" logits from the HuggingFace reference model for Kimi K2.

```sh
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \  
  --model-id=moonshotai/Kimi-K2-Instruct \  
  --prompts='I love to' \  
  --output-path=golden_Kimi-K2.jsonl \  
  --gcs-bucket=$my-gcs-bucket \  
  --hf-model-path=$LOCAL_BF16_PATH \  
  --hf-load-dtype=bfloat16 \  
  --trust-remote-code=True  
```

Run command below to compare logits between MaxText and HuggingFace.

```sh
JAX_PLATFORMS=cpu python3 -m tests.forward_pass_logit_checker \  
  src/maxtext/configs/base.yml \  
  base_output_directory=${BASE_OUTPUT_PATH?} \  
  run_name=forward_logits_check \  
  model_name=kimi-k2-1t \  
  load_parameters_path=${UNSCANNED_CKPT_PATH?} \  
  scan_layers=False \  
  async_checkpointing=False \  
  checkpoint_storage_concurrent_gb=1024 \  
  weight_dtype=bfloat16 \  
  ici_fsdp_parallelism=1 ici_expert_parallelism=-1 \  
  attention=dot_product \  
  per_device_batch_size=1 \  
  max_prefill_predict_length=4 max_target_length=4 \  
  sparse_matmul=False \  
  --golden_logits_path=${GOLDEN_LOGITS_DISK_LOCATION?} \  
  --atol=1.5 --rtol=1.5 --max_kl_div=0.1 \  
  --skip_first_token \  
  skip_jax_distributed_system=True
```

To run MMLU benchmarks and validate the model's performance, follow the instructions provided [here]( https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/api_server/README.md).

## Supported MoE strategy
* Dropless
  * [MegaBlocks](https://arxiv.org/abs/2211.15841) implementation with flag `sparse_matmul=True megablox=True`.
  * [JAX ragged_dot](https://github.com/jax-ml/jax/blob/a8fb0e01f8d083fff337d3c26375bb1b77344a99/jax/_src/lax/lax.py#L2415) implementation with flag `sparse_matmul=True megablox=False`.
  * General dense matmul implementation with flag `sparse_matmul=False capacity_factor=-1`.
* Dropping implementation with flag `sparse_matmul=False` and reasonable `capacity_factor`, commonly used from 1 to 1.25.