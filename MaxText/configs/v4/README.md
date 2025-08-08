<!--
 # SPDX-License-Identifier: Apache-2.0
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