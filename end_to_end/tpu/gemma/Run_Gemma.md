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

# Gemma
[Gemma](https://ai.google.dev/gemma) is a family of lightweight, state-of-the art open models built from research and technology that we used to create the Gemini models.

Following the instructions at [kaggle](https://www.kaggle.com/models/google/gemma/frameworks/maxText) will let you download Gemma model weights. You will have to consent to license for Gemma using your kaggle account's [API credentials](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials).

After downloading the weights run [convert_gemma_chkpt.py](../../MaxText/convert_gemma_chkpt.py), which converts the checkpoint to be compatible with MaxText and uploads them to a GCS bucket. You can run decode and finetuning using instructions mentioned in the test scripts at [end_to_end/tpu/gemma](../../end_to_end/tpu/gemma).

## MaxText supports pretraining and finetuning with high performance

Model Flop utilization for training on v5e and v5p TPUs.

| Model    | v5e-256 (bf16) | v5p-128 (bf16) | v5e-256 (int8) | v5p-128 (int8) |
| -------- | -------------- | -------------- | -------------- | -------------- |
| Gemma-2b | 58%            | 55%            | 64%            | 68%            |
| Gemma-7b | 58%            | 60%            | 70%            | 70%            |
