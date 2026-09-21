<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# m3 — Modern MaxText Models

m3 organizes models as a model garden: one self-contained directory per model family. A family
defines its own attention, MLP and decoder blocks instead of routing through a shared attention
class and a central `Decoder`, so a model can be read top to bottom in one place. Scalability
concerns are applied as post-construction transforms rather than branched on inside model code.

> **Status: scaffolding.** Model construction can opt in with `use_m3_model=true`, but the m3
> registry is currently empty, so every model is rejected. The flag defaults to false;
> legacy `src/maxtext/layers/` remains the production path until migration milestones complete.

## Layout

```text
m3/
  core/      # RoPE, sharding, interfaces, checkpoint, data — only what NNX lacks
  infra/     # remat, scan, quantization, offload — applied after construction
  configs/   # slim Pydantic config types + base.yml
  models/    # one directory per model family + create_model dispatch
  train/     # m3's own training loop (one of two supported drivers)
```

There is no `layers/` directory: NNX *is* the layer library.

## Architecture rules

1. **Use NNX built-ins directly.** Compose `nnx.Linear`, `nnx.Einsum`, `nnx.Embed`, `nnx.RMSNorm`,
   `nnx.Dropout` and `nnx.List` as-is — no wrapper layer classes, no Linen, no bridges.
2. **Models own their architecture.** Each model file defines its own attention, MLP and decoder
   block. No shared attention class parameterized for every variant, and no model-name branching.
   Some duplication across model files is cheaper than a 60-parameter shared layer.
3. **Sharding is an annotation.** Models receive a `Sharding` instance at construction and refer to
   logical axis names only (`"embed"`, `"heads"`, `"mlp"`). Model code never names physical mesh
   axes or devices.
4. **Infrastructure stays out of model code.** No `maybe_remat`, no `dot_general` injection, no
   `model_mode` conditionals. Only zero-cost annotations belong in `__call__`: sharding specs and
   `checkpoint_name` tags.
5. **Config-driven.** A typed Pydantic config selects architecture and hyperparameters; per-model
   YAML files override a base.
6. **Readability is a feature.** A model file should read top to bottom without jumping through a
   shared decoder.

## Invariants for the training-engine and RL path

m3 models must be drivable both by `m3/train/loop.py` and by the existing
`maxtext.training_engine.maxtext_engine` (which the RL stack targets). These invariants are load
bearing for that second driver — breaking them tends to fail silently:

| Invariant | Why |
| :---- | :---- |
| `nnx.Param` is the only trainable variable type | The engine differentiates `nnx.Param` only; other variable subclasses silently receive zero gradient |
| Do not `sow` intermediates on the engine path | Widening non-parameter state changes the state treedef across the forward pass and permanently drops the engine onto its slow path |
| Remat and scan wrappers must be path-transparent | Parameter paths feed checkpointing, HF mappings, weight sync, optimizer masks and unscan; an extra path segment breaks them, in several cases without an error |
| Auxiliary losses fold into the primary loss | Only the primary loss is differentiated; a router loss reported alongside it contributes no gradient |
| One `Mesh` object shared with the driver | Sharding whose mesh differs from the engine's is silently replaced by full replication |
| No parameter sharding constraints after init | Parameters are sharded once by the initializer; re-asserting collides with the engine's deferred all-reduce and Zero-1 resharding |

## Tests

Unit tests live in `tests/unit/m3/`, integration tests in `tests/integration/m3/`. They are picked
up by the standard MaxText pytest markers and run on every PR — no workflow changes are needed to
add a test file.
