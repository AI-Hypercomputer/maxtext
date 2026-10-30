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

> **Status: scaffolding.** Opt in with `use_m3_model=true`. `qwen3-0.6b` has been tested
> for correctness here, but performance features are not yet available.
> Legacy `src/maxtext/layers/` remains the production path
> until the migration milestones complete.

## Layout

```text
m3/
  core/      # RoPE, sharding, interfaces, checkpoint, data — only what NNX lacks
  infra/     # remat, scan, quantization, offload — applied after construction
  configs/   # slim Pydantic config types, model configs (e.g. glm5_3_flash_config) + base.yml
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
   axes or devices. Models are constructed inside `jax.sharding.set_mesh(mesh)`, on one device as
   on many: Flax places parameters from their metadata at construction and raises without a mesh.
4. **Infrastructure stays out of model code.** No `maybe_remat`, no `dot_general` injection, no
   `model_mode` conditionals. Only zero-cost annotations belong in `__call__`: sharding specs and
   `checkpoint_name` tags.
5. **Config-driven.** A typed Pydantic config selects architecture and hyperparameters; per-model
   YAML files override a base.
6. **Readability is a feature.** A model file should read top to bottom without jumping through a
   shared decoder.

## Precision and quantization

Precision follows legacy MaxText's mixed-precision pattern — float32 master weights and optimizer
moments, bfloat16 compute, float32 loss — and each concern lives in exactly one place:

| Concern | Where | Config |
| :---- | :---- | :---- |
| Parameter storage dtype (master weights) | `param_dtype=` on every NNX layer at construction | `weight_dtype` |
| Compute dtype | `dtype=` on every NNX layer at construction | `dtype` |
| Float32 islands (softmax, logits, routing, MoE combine) | the model's own arithmetic | — |
| Matmul precision for float32 operands | `jax.default_matmul_precision` around setup and the step (`train/loop.py`) | `matmul_precision` |
| int8 / fp8 matmuls | `infra.quantization.maybe_quantize`, after construction | `quantization` |

`matmul_precision` matters only when matmul operands are float32: on TPU `default` runs them as
one bfloat16 pass, `highest` as full float32. With bfloat16 compute it changes nothing, as in
legacy, where it is threaded into every `DenseGeneral`.

Quantization is a transform (`train/loop.py::setup_model`), applied with qwix by interception:
nothing in a model file changes, parameters stay `nnx.Param` with their paths, and the transform
asserts that. What it quantizes is decided structurally: every `nnx.Linear`, `nnx.LinearGeneral`
and `nnx.Einsum` under the decoder stack (`decoder/layers…`), except MoE routers (the projection
named `gate`). Each of those layers performs one weight matmul, and the qwix rule lists their
exact paths, so attention scores, the MoE combine and anything else a block computes in its own
`__call__` are never matched — nor are the embedding, the LM head or norms, which sit outside the
stack. The converse is checked too: a module under the stack that holds a weight without being one
of those layers is refused by name rather than left in full precision silently. Today that is
Qwen3-VL, whose language-model attention still uses a hand-rolled `DenseGeneral` (rule 1).

| `quantization=` | weights, activations | gradients |
| :---- | :---- | :---- |
| `int8` | int8 | int8 |
| `fp8` (also spelled `fp8_full`, legacy's name) | float8_e4m3fn | float8_e5m2 |
| `fp8_e4m3` (legacy's bare qwix `fp8`) | float8_e4m3fn | float8_e4m3fn |
| `fp8_e5m2` | float8_e5m2 | float8_e5m2 |

Scales are dynamic, calibration absmax. `quantization` is the only legacy quantization field m3
reads. A config that turns on a legacy knob m3 does not implement — `quantize_logits_proj`,
`quantize_mtp`, `sa_qproj_bwd_bf16`, a non-absmax calibration method, weight-gradient tiling,
sparsity, batch-split — is refused rather than trained with a recipe other than the one it names;
so is `grad_dtype` other than float32 in the loop. Two legacy fields are moot rather than refused:
`use_qwix_quantization` (m3 is qwix only) and `quantize_router_proj` (legacy quantizes the router
by default, m3 never does). Rather than run qwix's eager initialization pass, the transform
traces one forward pass of the training shape and requires one matmul in the recipe's dtype per
planned projection; a model with no projections under `decoder/layers` is refused, not trained
unquantized. Not yet: MoE grouped matmuls (they will join the plan through `ragged_dot`; the
Pallas executors need in-kernel quantization), fp8 attention (a kernel feature, not a qwix rule),
int4 / fp4, static scales, and weight-only quantization for serving. The legacy drivers
(`model_creation_utils.create_model`) do not apply m3 transforms yet.

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
| Mesh axes keep legacy's names (`data`, `fsdp`, `tensor`, `expert`) | The engine reads the data axis by name: on a mesh with no axis called `data` it declines Zero-1 (one warning) and deferred gradient all-reduce (silently) |
| No parameter sharding constraints after init | Parameters are sharded once by the initializer; re-asserting collides with the engine's deferred all-reduce and Zero-1 resharding |

## Tests

Unit tests live in `tests/unit/m3/`, integration tests in `tests/integration/m3/`. They are picked
up by the standard MaxText pytest markers and run on every PR — no workflow changes are needed to
add a test file.
