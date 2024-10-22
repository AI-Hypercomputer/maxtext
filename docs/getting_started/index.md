# Getting Started

For your first time running MaxText, we provide specific [instructions](First_run.md).

MaxText supports training and inference of various open models.

Some extra helpful guides:
* [Gemma](https://ai.google.dev/gemma): a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/), based on Gemini research and technology. You can run decode and finetuning using [these instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gemma/Run_Gemma.md).
* [Llama2](https://llama.meta.com/llama2/): a family of open-weights Large Language Model (LLM) by Meta. You can run decode and finetuning using [these instructions](Run_Llama2.md).
* [Mixtral](https://mistral.ai/news/mixtral-of-experts/): a family of open-weights sparse mixture-of-experts (MoE) model by Mistral AI. You can run decode and finetuning using [these instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/Run_Mixtral.md)

In addition to the getting started guides, there are always other MaxText capabilities that are being constantly being added! The full suite of end-to-end tests is in [end_to_end](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end). We run them with a nightly cadence. They can be a good source for understanding MaxText Alternatively you can see the continuous [unit tests](https://github.com/AI-Hypercomputer/maxtext/blob/main/.github/workflows/UnitTests.yml) which are run almost continuously.

```{toctree}
:hidden:

First_run.md
steps_model.md
End-to-end example <https://www.kaggle.com/code/melissawm/maxtext-examples>
```
