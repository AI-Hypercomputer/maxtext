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

# High Performance Model Configs on A3 GPU
Expected performance results for Llama2-7B model running on A3 GPU:


### Llama2-7B
| Hardware               | TFLOP/sec/chip   |
| ---------------------- | ---------------- |
| 1x A3 (h100-80gb-8)    | 492              |
| 2x A3 (h100-80gb-8)    | 422              |
| 4x A3 (h100-80gb-8)    | 407              |
| 8x A3 (h100-80gb-8)    | 409              |
| 16x A3 (h100-80gb-8)   | 375              |
