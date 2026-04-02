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

# Gemma4

Gemma is a family of open models built by Google DeepMind. [Gemma 4](https://ai.google.dev/gemma) models are multimodal, handling text and image input and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants, featuring a context window of up to 256K tokens and multilingual support in over 140 languages.

Gemma 4 in MaxText is available in two sizes with Dense (31B) and Mixture-of-Experts (MoE) (26B A4B) architectures, and is well-suited for tasks like text generation, coding, and reasoning. The models are designed for enhanced performance and efficiency, capable of running on environments ranging from laptops and servers.

We provide examples for checkpoint conversion scripts at [tests/end_to_end/tpu/gemma4](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu/gemma4).

## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining Gemma4-26B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_path=${DATASET_PATH?} tokenizer_path=google/gemma-4-26b-a4b-it per_device_batch_size=1 run_name=runner_pretrain_gemma4_26b steps=10 enable_checkpointing=false sharding_tolerance=0.03
```

## Checkpoint Conversion
To obtain the Gemma4 model weights, you can access them on Hugging Face (e.g., [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)). You will need to accept the Gemma4 license through your Hugging Face account and provide your Hugging Face access token (as `HF_TOKEN`) for authentication. You can then convert them directly into a MaxText compatible format. Here's an example of converting the model weights using the conversion script (`tests/end_to_end/tpu/gemma4/26b/convert_gemma4_26b.sh`):

```sh
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=gemma4-26b \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET}/26b/converted/${idx} \
    use_multimodal=true \
    scan_layers=false
```

This will convert the checkpoints and save them to a Google Cloud Storage (GCS) bucket.

## Fine-tuning
After the conversion, you will have a MaxText compatible checkpoint which allows you to fine-tune it with different datasets. For more comprehensive guides, please refer to our tutorials on [Multimodal Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/multimodal.md#supervised-fine-tuning) and [Supervised Fine-Tuning](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/posttraining/sft.md). One example command to fine-tune a Gemma4-26B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma4-26b base_output_directory=${BASE_OUTPUT_DIRECTORY?} dataset_type=synthetic tokenizer_type=huggingface load_parameters_path=${CONVERTED_CHECKPOINT?} tokenizer_path=google/gemma-4-26b-a4b-it per_device_batch_size=1 run_name=runner_finetune_gemma4_26b steps=10 enable_checkpointing=true sharding_tolerance=0.03
```

## Inference
For detailed instructions on running inference and decoding with MaxText, please refer to our [Inference Tutorial](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/inference.md).
