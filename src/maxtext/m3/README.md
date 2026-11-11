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

For the motivation, design rationale, and alternatives considered, see the
[m3 design doc](DESIGN.md).

## Layout

```text
m3/
  core/      # RoPE, sharding, interfaces, checkpoint, data — only what NNX lacks
  infra/     # remat, scan, quantization, offload — applied after construction
  configs/   # slim Pydantic config types, model configs (e.g. glm5_3_flash_config) + base.yml
  models/    # one directory per model family + create_model dispatch
  train/     # m3's own training loop (one of two supported drivers)
  diffuser_pipelines/  # diffusion model inference pipelines: checkpoint loading, denoising loop, VAE decode; driven by inference/sample.py
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
and `nnx.Einsum` under the decoder stack (`decoder/layers…`), including MoE routers (anything under
a `gate` module) unless `quantize_router_proj=false`, as in legacy. Each of those layers performs one weight matmul, and the qwix rule lists their
exact paths, so attention scores, the MoE combine and anything else a block computes in its own
`__call__` are never matched — nor are the embedding, the LM head or norms, which sit outside the
stack. The converse is checked too: a module under the stack that holds a weight without being one
of those layers is refused by name rather than left in full precision silently. Today that is
Qwen3-VL, whose language-model attention still uses a hand-rolled `DenseGeneral` (rule 1).

| `quantization=` | weights, activations | gradients |
| :---- | :---- | :---- |
| `int8` | int8 | int8 |
| `fp8`, `fp8_e4m3` | float8_e4m3fn | float8_e4m3fn |
| `fp8_e5m2` | float8_e5m2 | float8_e5m2 |
| `fp8_full` | float8_e4m3fn | float8_e5m2 |

The names mean what they mean in legacy's qwix path. Quantization requires
`use_qwix_quantization=true`: with it false, legacy trains the same config with AQT, which m3
doesn't have, so m3 refuses rather than switching recipe.

Scales are dynamic, calibration absmax. `quantization` and `quantize_router_proj` are the legacy
quantization fields m3 reads. A config that turns on a legacy knob m3 does not implement — `quantize_logits_proj`,
`quantize_mtp`, `sa_qproj_bwd_bf16`, a non-absmax calibration method, weight-gradient tiling,
block-scaled weights (`weight_block_size`), `unquantized_modules`, sparsity, batch-split — is refused rather than trained with a recipe other than the one it names;
so is `grad_dtype` other than float32 in the loop. Rather than run qwix's eager initialization pass, the transform
traces one forward pass of the training shape and requires one matmul in the recipe's dtype per
planned projection; a model with no projections under `decoder/layers` is refused, not trained
unquantized. Not yet: MoE grouped matmuls (they will join the plan through `ragged_dot`; the
Pallas executors need in-kernel quantization), fp8 attention (a kernel feature, not a qwix rule),
int4 / fp4, static scales, and weight-only quantization for serving. The legacy drivers
(`model_creation_utils.create_model`) do not apply m3 transforms yet.

## Sharding

Rule 3 end to end. A model declares sharding once, in logical axis names, at construction; a
`Sharding` strategy resolves them; the mesh decides which axes are live. Nothing is decided by
path regexes or heuristics, and nothing re-shards a parameter after it is created.

| Concern | Where | Config |
| :---- | :---- | :---- |
| Which logical axis each tensor dimension carries | the model, as `*_metadata={"out_sharding": s(name, axes)}` on every parameter and `s.constrain(x, name, axes)` on key activations | — |
| Which mesh axis a logical axis lands on | one strategy class per parallelism scheme: `core.sharding.DefaultSharding` (`fsdp` × `fsdp_transpose` × `context` × `expert` over `embed`, where `fsdp_transpose` is an extra FSDP factor; `tensor` and `tensor_sequence` over `heads`, `kv_heads`, `mlp`, `vocab`; `context` over `length`; `context` and `tensor_sequence` over `norm_length`, using `tensor_sequence` instead of `tensor` for Megatron SP; batch over `data` × `fsdp` × `fsdp_transpose` × `expert`) | — |
| Mesh shape | `core.sharding.mesh_from_config` → `create_mesh` | `ici_{data,fsdp,fsdp_transpose,context,tensor,tensor_sequence,expert}_parallelism`, `dcn_*`, `custom_mesh` (`hybrid_ring_64x4`, `hybrid_ring_32x8`), `allow_split_physical_axes`, `optimize_mesh_for_tpu_v6e` |
| Auto vs explicit sharding | `create_mesh(explicit=)`; `constrain` and `out_sharding` dispatch on the mesh in scope | `shard_mode` (`auto`, the default, or `explicit` on models that wire `out_sharding` — `qwen3` today) |
| Input batch layout | the driver asks the strategy (`train/loop.py::batch_spec`) | — |

Parameters are placed by Flax from their metadata the moment they are created, under whatever mesh
is in scope, so single-device tests and an 8-chip run construct the same way. That placement is
declared per parameter (`core.sharding.param_metadata` sets `eager_sharding=True` next to
`out_sharding`) rather than left to Flax's process-global switch, which legacy MaxText turns off at
import (`layers/nnx_wrappers.py`) and which would otherwise leave every m3 parameter replicated
without a word. The flag is for construction only: `models.create_model` turns it off once the
model is built, because Flax rebuilds variables from their metadata (`nnx.Optimizer`'s moments, for
one) and would otherwise re-run placement, which fails with no mesh in scope. Both drivers (the m3
loop and the training engine) read placement off the arrays. The loop logs the fraction of
parameters each device holds; `100%` means replicated, which is what an unannotated model gets.

`shard_mode=explicit` builds an Explicit-typed mesh, passes explicit `out_sharding` specs to NNX
embedding gathers and linear/einsum projections, and lowers the scalar loss reduction via
`jax.sharding.auto_axes`, enabling `MaxTextTrainingEngine` ZeRO-1 optimizer state sharding on
models that implement `out_sharding` (`qwen3` today; other registered models refuse
`shard_mode=explicit` with a named error, `quantization` requires `shard_mode=auto` until upstream
`qwix` forwards `out_sharding`, and pure-DP deferred gradient reductions remain disabled on m3
because stock `nnx.RMSNorm` does not accept `reduced`-tagged scale operands).

A model with a novel axis subclasses `DefaultSharding` and delegates the rest; a driver that wants
another scheme passes `sharding=` to `create_model` (or `sharding_from_config(config)`). m3 does not
read `logical_axis_rules`: the strategy class is the rule table, typed and conflict-checked. Legacy
parallelism settings m3 does not implement (`pipeline`, `sequence`, `context_usp_ulysses`,
`context_autoregressive`, `autoregressive`, `diloco`, `custom_mesh_and_rule*`, non-`all_gather`
`context_parallel_strategy`, and `context_parallel_load_balance=True` when `context > 1`) and mesh
axes a selected model's `Sharding` does not partition are refused with the field named rather than
ignored.

Today `qwen3` is fully annotated; Qwen3-VL inherits the decoder, MLP and embedding annotations but
its hand-rolled attention and vision tower are not yet sharded; GLM-5.3-Flash and Wan carry their
own `Sharding` subclasses and mesh construction, which should converge on `DefaultSharding` and
`mesh_from_config`.

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
