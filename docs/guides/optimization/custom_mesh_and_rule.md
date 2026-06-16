<!--
 Copyright 2026 Google LLC

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

# Customizing mesh and rule for advanced sharding features in MaxText

## How `custom_mesh_and_rule` Works

In MaxText, the `custom_mesh_and_rule` configuration allows you to completely override the default device mesh and logical axis rules used for sharding. Instead of relying on the standard rules defined in the main configuration (`base.yml`), you can point this parameter to a specific YAML file located in the `src/maxtext/configs/custom_mesh_and_rule/` directory.

When you specify a rule name (e.g., `custom_mesh_and_rule=pure-fsdp`), MaxText loads the corresponding YAML file and applies its specific:

- `mesh_axes`: The physical layout of the devices.
- `data_sharding`: Axes used for sharding input data.
- `context_sharding`: Which physical axis plays the role of context parallelism.
- `logical_axis_rules`: The precise mapping of logical tensor axes to physical device mesh axes.

## When to Use Custom Meshes and Rules

While MaxText's default sharding strategies handle most standard models and configurations effectively, you may need to define custom meshes and rules in the following scenarios:

- **Simplifying logical rules:** If you are training on a smaller cluster where only one or two axes are necessary, a simplified, custom logical rule focusing exclusively on those axes can significantly streamline the sharding debugging process.
- **Managing large-scale training:** At scale, you must carefully dictate how specific tensors are sharded or replicated to respect HBM and sharding dimension limits. For example, if a `q_lora` tensor with a dimension of 512 is sharded across FSDP, Expert, and Context axes, an error will occur if the product of these axes exceeds 512. A custom rule allows you to drop conflicting axes to prevent these dimension overflows.
- **Implementing advanced sharding features:** To maximize performance, you might want to repurpose specific axes dynamically based on the layer. For instance, you could configure the Expert axis to handle Context parallelism outside of Mixture of Experts (MoE) layers. Achieving this level of granular flexibility requires custom sharding rules.

## Pre-Defined Sharding Configurations

MaxText currently provides several ready-to-use custom mesh and rule configurations:

### `pure-fsdp.yml`

This rule relies entirely on Fully Sharded Data Parallelism (FSDP). It maps all activations and weights directly to the `fsdp` mesh axis. This is the recommended sharding strategy for small-scale training, as it simplifies the overall configuration and makes debugging significantly easier.

### `ep-as-cp.yml`

This rule utilizes the `data`, `stage`, `fsdp`, and `expert` axes. Its defining feature is that it repurposes the `expert` axis to handle context parallelism in all components *except* for the core dense Mixture of Experts (MoE) layers (i.e., the computations between Expert Parallelism all-to-all communications). By reusing the expert dimension to shard the sequence length in non-MoE layers, it enables fractional batch size to reduce HBM usage.

### `cp-as-ep.yml`

Similar in philosophy to `ep-as-cp.yml`, this configuration explicitly includes the `context` axis in the mesh layout alongside `data`, `stage`, `fsdp`, and `expert`. While context sharding is mapped to the `context` axis globally, within MoE components, this `context` axis dynamically shifts to perform expert parallelism instead of FSDP. This custom rule supports using CP and EP together.

### `pipeline-large-moe.yml`

Designed specifically to optimize pipeline parallelism for extremely large-scale MoE jobs (such as DeepSeek models). It defines the physical axes: `data`, `stage`, `fsdp`, `tensor`, `context`, and `expert`. To prevent dimension limit errors, it intentionally disables expert weight sharding on the (typically small) `q_lora` dimension. Furthermore, tensor and expert parallelism are strictly preserved to support advanced pipelining features like `pipeline_fsdp_ag_one` and `pipeline_fsdp_ag_per_repeat`.

## Protecting Configurations with Sharding Dump

Because custom sharding rules are highly specific and sensitive to changes in model architecture, MaxText uses an automated **Sharding Dump** mechanism to protect them against regressions.

1. **Dumping State:** The `tests/utils/sharding_dump.py` tool generates an abstract state representation of how tensors are sharded across the mesh for a given model, topology, and `custom_mesh_and_rule`. It outputs this layout into a JSON file.
2. **Comparison Testing:** In `tests/unit/sharding_compare_test.py`, test cases instantiate models with specific custom mesh rules and compare the generated sharding JSON against known-good "golden" files stored in `tests/utils/sharding_info/`.
3. **Regression Prevention:** If a code change inadvertently alters the sharding layout (for example, causing an activation to un-shard and run out of memory), the comparison test will fail, alerting developers before the issue affects actual large-scale training jobs.
