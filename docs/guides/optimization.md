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

# Optimization

Explore techniques for maximizing performance, including model customization, sharding strategies, Pallas kernels, and benchmarking.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 🛠️ Customizing Model Configs
:link: optimization/custom_model
:link-type: doc

Optimize and customize your LLM model configurations for higher performance (MFU) on TPUs.
:::

:::{grid-item-card} 🥞 Sharding Strategies
:link: optimization/sharding
:link-type: doc

Choose efficient sharding strategies (FSDP, TP, EP, PP) using Roofline Analysis and understand arithmetic intensity.
:::

:::{grid-item-card} ⚡ Pallas Kernels
:link: optimization/pallas_kernels_performance
:link-type: doc

Optimize with Pallas kernels for fine-grained control. 
:::

:::{grid-item-card} 📈 Benchmarking & Tuning
:link: optimization/benchmark_and_performance
:link-type: doc

Guide to setting up benchmarks, performing performance tuning, and analyzing metrics.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

optimization/custom_model.md
optimization/sharding.md
optimization/pallas_kernels_performance.md
optimization/benchmark_and_performance.md
```
