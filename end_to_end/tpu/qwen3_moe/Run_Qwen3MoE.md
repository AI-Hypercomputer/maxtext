<!--
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Qwen3 MoE

Qwen3 MoE is Alibaba's mixture-of-experts model family. This example shows how to
convert the HuggingFace checkpoints for use with MaxText and run a small
inference job on TPU.

The workflow consists of two steps:
1. Use `convert_qwen3_moe_ckpt.py` to convert the HuggingFace weights to MaxText
   Orbax format using a CPU VM.
2. Run `forward_pass_logit_checker.py` on a TPU VM with the converted weights.

Scripts `1_test_qwen3_moe.sh` and `2_test_qwen3_moe.sh` in this directory
provide sample commands for these two steps.

To export a fine-tuned checkpoint back to HuggingFace format see
`convert_qwen3_moe_to_hf.sh` in `utils/ckpt_conversion/examples` which wraps
`MaxText.ckpt_conversion.to_huggingface` for the Qwen3 MoE model.

