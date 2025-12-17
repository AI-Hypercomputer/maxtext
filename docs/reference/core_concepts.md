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

# Core concepts

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 💾 Checkpoints
:link: core_concepts/checkpoints
:link-type: doc

Understanding checkpoint formats and strategies.
:::

:::{grid-item-card} ⚖️ Alternatives
:link: core_concepts/alternatives
:link-type: doc

Comparison with other frameworks like Megatron-LM.
:::

:::{grid-item-card} 📉 Quantization
:link: core_concepts/quantization
:link-type: doc

Techniques for reducing model size and improving performance.
:::

:::{grid-item-card} 🧱 Tiling
:link: core_concepts/tiling
:link-type: doc

Understanding tiling strategies for partitioning logic.
:::

:::{grid-item-card} ⚡ JAX/XLA/Pallas
:link: core_concepts/jax_xla_and_pallas
:link-type: doc

How MaxText leverages JAX, XLA, and Pallas for efficiency.
:::

:::{grid-item-card} 🧠 MoE Configuration
:link: core_concepts/moe_configuration
:link-type: doc

Configuring Mixture of Experts (MoE) models.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

core_concepts/checkpoints.md
core_concepts/alternatives.md
core_concepts/quantization.md
core_concepts/tiling.md
core_concepts/jax_xla_and_pallas.md
core_concepts/moe_configuration.md
```
