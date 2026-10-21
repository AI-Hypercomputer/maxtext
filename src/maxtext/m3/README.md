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

> **Status: draft of the full shape.** Opt in with `use_m3_model=true`. Qwen3 (`qwen3-0.6b`),
> Qwen3-MoE (`qwen3-30b-a3b`, `qwen3-235b-a22b`) and DeepSeek-V3 (`deepseek3-671b`,
> `deepseek3-tiny`) have legacy parity tests; Qwen3-VL has path parity. Remat, scan, qwix
> quantization and optimizer host offload exist as transforms and compose in `train/loop.py`.
> Fused kernels -- splash / tokamax attention, megablox / tokamax grouped matmul -- are selected
> from the legacy config fields and sharded inside `core/kernels.py`.
> Not yet: capacity-factor token dropping, expert parallelism for the Pallas MoE executors,
> context parallelism, DeepSeek's loss-free bias updates, MTP, KV caches, checkpointing, real data.
> Legacy `src/maxtext/layers/` remains the production path.

## Layout

```text
m3/
  core/      # RoPE, attention fn, kernel adapters, sharding, Layers container, MoE runtime, CausalLMOutput
  infra/     # remat, scan, quantization, offload — applied after construction
  configs/   # slim Pydantic config types + base.yml (planned; legacy pyconfig is used today)
  models/    # one directory per model family + create_model dispatch
  train/     # loss and m3's own training loop (one of two supported drivers)
```

There is no `layers/` directory: NNX *is* the layer library.

## How a model is put together

```python
with jax.set_mesh(create_mesh()):
  model = models.create_model(config, mesh, rngs=nnx.Rngs(0))   # Sharding resolves logical axes
  model = remat.maybe_apply_remat(model, config)                # sets Layers.remat; no wrapper
  model = scan.maybe_apply_scan(model, config)                  # Layers -> ScannedLayers
  model = quantization.maybe_quantize(model, config, *batch)    # qwix, by interception
  optimizer = offload.make_optimizer(model, tx, config)         # state on pinned_host if asked
loss, parts = train.loss.loss_fn(model, batch)                  # cross entropy + model.aux_loss
```

Every m3 model's `__call__` returns a `CausalLMOutput(logits, aux_loss)`. MoE models carry their
load-balance loss through the decoder stack as part of the layer carry and return it there; nothing
is `sow`n.

### Kernels

Fused kernels are not a transform and not a model concern. A model resolves a kernel object once,
at construction, from the config and its `Sharding`, and passes it to the one arithmetic function it
already calls:

```python
self.attention_kernel = kernels.attention_from_config(config, sharding)   # in __init__
out = dot_product_attention(q, k, v, positions=..., segment_ids=..., kernel=self.attention_kernel)

self.executor = moe.executor_from_config(config, sharding)                # MoE blocks
out = moe.routed_experts(x, weights, indices, wi_0, wi_1, wo, executor=self.executor)
```

The kernel object is a frozen dataclass holding the backend name and the mesh axes the kernel's
inputs are sharded over (looked up through the `Sharding`, so model code still names logical axes
only). Everything a Pallas kernel needs that XLA cannot do for it -- the `shard_map`, block sizes,
the FSDP all-gather of expert weights, the reduction over a tensor-sharded MLP dimension -- lives in
`core/kernels.py` and `core/moe.py`. Backends come from upstream JAX and `tokamax`, not from the
forked copies under `src/maxtext/kernels/`. Legacy config names are translated in one place each:
`attention=dot_product|flash|autoselected` (+ `use_tokamax_splash`, `sa_block_*`) and
`sparse_matmul` / `megablox` / `use_tokamax_gmm`.


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
