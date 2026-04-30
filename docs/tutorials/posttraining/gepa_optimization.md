# GEPA Prompt Optimization for MaxText

## Overview

This document explains how to use **GEPA** (Generic Evaluation and Prompt Adaptation) to optimize system prompts for MaxText models. GEPA is an evolutionary framework ([GitHub Repository](https://github.com/gepa-ai/gepa), [Paper](https://arxiv.org/abs/2507.19457)) that iteratively refines prompts based on evaluation feedback, helping models perform better on specific tasks. A complete, runnable example notebook is provided in the repository at [maxtext_with_gepa.ipynb](../../../src/maxtext/examples/maxtext_with_gepa.ipynb).

## How GEPA Optimization Works

The optimization process relies on a collaborative loop between two Language Models (LMs):

1. **Target Model**: This is the model being optimized. It attempts to solve the evaluation problems (e.g., AIME questions) using the current candidate system prompt. For example, this can be a `Qwen3-4B` model hosted on a local vLLM server.
2. **Reflection LM**: This model reviews the reasoning traces and failures of the Target Model. It identifies recurring errors (e.g., mathematical errors or formatting issues) and proposes targeted updates to the system prompt. For example, a model like `Gemini 3 Flash Preview` can be used as the reflection model.

### The Evolutionary Loop

1. **Propose**: The Reflection LM proposes a new system prompt based on errors seen in previous runs.
2. **Evaluate (Subsample)**: The Target Model solves a small random subset of problems using the new prompt. This serves as a quick screening step.
3. **Full Evaluation**: If the subsample score improves, the prompt is evaluated on the full validation set.
4. **Selection**: Successful prompts are added to the candidate pool, driving the evolution of domain-specific heuristics (such as circle packing formulas or prime factorization strategies) that eventually form the final optimized prompt.

### Synergy via Prompt Merging

A key feature used during the AIME experimentation was **Prompt Merging** (`use_merge=True`).

As the evolutionary process runs, different branches might discover distinct, valid heuristics (e.g., one branch learns a rule for Geometry, while another learns a rule for Combinatorics).

- **How It Works**: Instead of forcing a choice between these two distinct winning paths, GEPA attempts to merge them. The Reflection LM is instructed to synthesize the instructions from both candidates, deduplicating content and integrating the new knowledge into a single, unified system prompt.
- **Why It Is Important**: Merging allows the optimization to achieve synergetic gains. By combining orthogonal prompt improvements, the final system prompt acts as a comprehensive "cheat sheet" covering multiple mathematical domains simultaneously, which is critical for the broad range of problems found in datasets like AIME.

## Robust Evaluation with MathAdapter

A critical component of the optimization setup is the custom `MathAdapter`.

### Why the Custom Logic?

Standard evaluation pipelines often use simple regular expressions to extract the answer from a model's response (e.g., capturing everything inside `\boxed{}`). However, competition math problems like AIME frequently require answers formatted in complex LaTeX (such as fractions `\boxed{\frac{a}{b}}` or nested expressions). A naive regex will break on the first closing brace `}`, failing to capture the full answer.

The `MathAdapter` implements a robust **brace-counting parser** that correctly tracks nested LaTeX structures, ensuring the complete mathematical expression is extracted.

### Why It Is Crucial for GEPA

Prompt optimization frameworks like GEPA are highly sensitive to the reward signal (the evaluation score). If a model generates a correct answer but the evaluation logic fails to parse it correctly (a False Negative), the optimization loop receives faulty feedback. This noisy signal can cause GEPA to discard beneficial prompt mutations, ultimately leading to performance degradation instead of improvement.

## Tutorial Notebook

A complete, runnable tutorial is available in the repository as a Jupyter Notebook:
[maxtext_with_gepa.ipynb](../../../src/maxtext/examples/maxtext_with_gepa.ipynb) (provided as an example)

This notebook walks through:

- Streaming the dataset.
- Setting up a custom `MathAdapter` for float extraction.
- Running the GEPA evolutionary loop.
- Comparing accuracy before and after optimization.

> [!NOTE]
> In this tutorial, we utilize an out-of-tree version of vLLM tailored for MaxText models via the `maxtext_vllm_adapter`. For more information on serving MaxText models with vLLM, refer to the [Inference Guide](../inference.md).

## Pointing GEPA to the Local vLLM Server

By default, optimization frameworks might expect to communicate with remote model APIs. In our setup, we route the evaluation traffic to the locally running MaxText model on the vLLM server by overriding the API base URL.

This is achieved by setting the following environment variables in the script:

```python
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "fake-key"
```

When the `MathAdapter` initializes the model (e.g., specifying `openai/Qwen/Qwen3-4B-Instruct-2507`), `litellm` (used by GEPA under the hood) intercepts the request and directs it to the local server running on the TPU host instead of attempting to connect to a remote OpenAI endpoint.

## Case Study: AIME Prompt Optimization

In our experiments with the **AIME (American Invitational Mathematics Examination)** dataset, we utilized **Qwen3-4B** as the Target Model (hosted locally via vLLM) and **Gemini 3 Flash Preview** as the Reflection LM.

With this setup, GEPA successfully improved the model's accuracy from **49.0% to 54.0%** (a 5% absolute improvement).

The optimization process discovered that injecting specific domain knowledge and heuristics (like circle packing formulas and square-free parts for number theory) significantly helped the model solve complex competition-level problems.
