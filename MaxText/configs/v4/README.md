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

# High Performance Model Configs on TPU v4
Expected performance results for 22B and 52B parameter models running on TPU v4:


### 22B model
| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 1x v4-128   | 156              | 56.7% |
| 2x v4-128   | 152              | 55.2% |
| 4x v4-128   | 149              | 54.3% |
| 8x v4-128   | 146              | 53.2% |

### 52B model
| Hardware    | TFLOP/sec/chip   |  MFU  |
| ----------- | ---------------- | ----- |
| 1x v4-384   | 154              | 56.0% |
| 2x v4-384   | 162              | 58.9% | # this is quirkily higher than single slice because of choices made by the compiler, not for a fundamental reason.