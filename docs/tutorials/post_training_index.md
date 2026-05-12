# Post-training

```{note}
Post-training workflows on TPU require specific dependencies. Please ensure you have installed MaxText with `maxtext[tpu-post-train]` as described in the [official documentation](https://maxtext.readthedocs.io/en/latest/install_maxtext.html).
```

## What is MaxText post-training?

MaxText provides performance and scalable LLM and VLM post-training, across a variety of techniques like SFT and GRPO.

We’re investing in performance, scale, algorithms, models, reliability, and ease of use to provide the most competitive OSS solution available.

## The MaxText stack

MaxText was co-designed with key Google led innovations to provide a unified post training experience:

- [MaxText model library](supported-model-families) for JAX LLMs highly optimized for TPUs
- [Tunix](https://github.com/google/tunix) for the latest algorithms and post-training techniques
- [vLLM on TPU](https://github.com/vllm-project/tpu-inference) for high performance sampling (inference) for Reinforcement Learning (RL)
- [Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro) for multi-host inference (sampling) and highly efficient weight transfer

![GRPO Diagram](../_static/grpo_diagram.png)

## Supported techniques & models

- **SFT (Supervised Fine-Tuning)**
  - [SFT on Single-Host TPUs](./posttraining/sft)
  - [SFT on Multi-Host TPUs](./posttraining/sft_on_multi_host)
- **Multimodal SFT**
  - [Multimodal Support](./posttraining/multimodal)
- **Reinforcement Learning (RL)**
  - [RL on Single-Host TPUs](./posttraining/rl)
  - [RL on Multi-Host TPUs](./posttraining/rl_on_multi_host)

## Step by step RL

Making powerful RL accessible is at the core of the MaxText mission

Here is an example of the steps you might go through to run a Reinforcement Learning (RL) job:

![RL Workflow](../_static/rl_workflow.png)

## What is Pathways and why is it key for RL?

Pathways is a single controller JAX runtime that was [designed and pressure tested internally at Google DeepMind](https://blog.google/innovation-and-ai/products/introducing-pathways-next-generation-ai-architecture/) over many years. Now available on Google Cloud, it is designed to coordinate distributed computations across thousands of accelerators from a single Python program. It efficiently performs data transfers between accelerators both within a slice using ICI (Inter-chip Interconnect) and across slices over DCN (Data Center Network).

Pathways allows for fine grained resource allocation (subslice of a physical slice) and scheduling. This allows JAX developers to explore novel model architectures in an easy to develop single controller programming environment.

Pathways supercharges RL with:

1. **Multi-host Model Support:** Easily manages models that span multiple hosts.
2. **Unified Orchestration:** Controls both trainers and samplers from a single Python process.
3. **Efficient Data Transfer:** Optimally moves data between training and inference devices, utilizing ICI or DCN as needed. JAX Reshard primitives simplify integration.
4. **Flexible Resource Allocation:** Enables dedicating different numbers of accelerators to inference and training within the same job, adapting to workload bottlenecks (disaggregated setup).

## Getting started

Start your Post-Training journey through quick experimentation with [Python Notebooks](../guides/run_python_notebook) or our Production level tutorials for [SFT](./posttraining/sft_on_multi_host) and [RL](./posttraining/rl_on_multi_host).

## More tutorials

```{toctree}
---
maxdepth: 1
---
posttraining/sft.md
posttraining/sft_on_multi_host.md
posttraining/rl.md
posttraining/rl_on_multi_host.md
posttraining/knowledge_distillation.md
posttraining/multimodal.md
posttraining/full_finetuning.md
posttraining/gepa_optimization.md
```
