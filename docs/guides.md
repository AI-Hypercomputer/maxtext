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

# How-to guides

Explore our how-to guides for optimizing, debugging, and managing your MaxText workloads.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} ⚡ Optimization
:link: guides/optimization
:link-type: doc

Techniques for maximizing performance, including sharding strategies, Pallas kernels, and benchmarking.
:::

:::{grid-item-card} 💾 Data Pipelines
:link: guides/data_input_pipeline
:link-type: doc

Configure input pipelines using **Grain** (recommended for determinism), **HuggingFace**, or **TFDS**.
:::

:::{grid-item-card} 🔄 Checkpointing
:link: guides/checkpointing_solutions
:link-type: doc

Manage GCS checkpoints, handle preemption with emergency checkpointing, and configure multi-tier storage.
:::

:::{grid-item-card} 🔍 Monitoring & Debugging
:link: guides/monitoring_and_debugging
:link-type: doc

Tools for observability: goodput monitoring, hung job debugging, and Vertex AI TensorBoard integration.
:::

:::{grid-item-card} 🐍 Python Notebooks
:link: guides/run_python_notebook
:link-type: doc

Interactive development guides for running MaxText on Google Colab or local JupyterLab environments.
:::

:::{grid-item-card} 🌱 Model Bringup
:link: guides/model_bringup
:link-type: doc

A step-by-step guide for the community to help expand MaxText's model library.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

guides/optimization.md
guides/data_input_pipeline.md
guides/checkpointing_solutions.md
guides/monitoring_and_debugging.md
guides/run_python_notebook.md
guides/model_bringup.md
```
