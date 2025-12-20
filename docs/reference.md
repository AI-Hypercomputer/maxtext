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

# Reference documentation

Deep dive into MaxText architecture, models, and core concepts.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 📊 Performance Metrics
:link: reference/performance_metrics
:link-type: doc

Understanding Model Flops Utilization (MFU), calculation methods, and why it matters for performance optimization.
:::

:::{grid-item-card} 🤖 Models
:link: reference/models
:link-type: doc

Supported models and architectures, including Llama, Qwen, and Mixtral. Details on tiering and new additions.
:::

:::{grid-item-card} 🏗️ Architecture
:link: reference/architecture
:link-type: doc

High-level overview of MaxText design, JAX/XLA choices, and how components interact.
:::

:::{grid-item-card} 💡 Core Concepts
:link: reference/core_concepts
:link-type: doc

Key concepts including checkpointing strategies, quantization, tiling, and Mixture of Experts (MoE) configuration.
:::
::::

## 📚 API Reference

Find comprehensive API documentation for MaxText modules, classes, and functions in the [API Reference page](reference/api_reference).


```{toctree}
:hidden:
:maxdepth: 1

reference/performance_metrics.md
reference/models.md
reference/architecture.md
reference/core_concepts.md
reference/api_reference.rst
```
