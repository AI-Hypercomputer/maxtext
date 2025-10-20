<!--
 Copyright 2024 Google LLC

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

# Tutorials

For your first time running MaxText, we provide specific [instructions](first-run).

MaxText supports training and inference of various open models.

Some extra helpful guides:
* [Gemma](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. You can run decode and finetuning using [these instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gemma/Run_Gemma.md).
* [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. You can run decode and finetuning using [these instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama2/run_llama2.md).
* [Mixtral](https://mistral.ai/news/mixtral-of-experts/): a family of open-weights sparse mixture-of-experts (MoE) model by Mistral AI. You can run decode and finetuning using [these instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/Run_Mixtral.md)

In addition to the getting started guides, there are always other MaxText capabilities that are being constantly being added! The full suite of end-to-end tests is in [end_to_end](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end). We run them with a nightly cadence. They can be a good source for understanding MaxText Alternatively you can see the continuous [unit tests](https://github.com/AI-Hypercomputer/maxtext/blob/main/.github/workflows/RunTests.yml) which are run almost continuously.

## End-to-end example

See the <a href="https://www.kaggle.com/code/shivajidutta/maxtext-on-kaggle" target="_blank">MaxText example Kaggle notebook</a>.

## Other examples

You can also find other examples in the [MaxText repository](https://github.com/AI-Hypercomputer/maxtext/tree/main/pedagogical_examples).

```{toctree}
:maxdepth: 1

tutorials/first_run.md
tutorials/pretraining.md
tutorials/full_finetuning.md
tutorials/grpo.md
tutorials/sft.md
tutorials/sft_on_multi_host.md
tutorials/grpo_with_pathways.md
```
