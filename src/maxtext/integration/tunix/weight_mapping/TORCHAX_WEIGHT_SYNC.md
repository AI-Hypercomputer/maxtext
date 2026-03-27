# MaxText NNX → torchax Weight Sync

## Context

`bench_weight_sync.py` converts MaxText NNX weights to vLLM/torchax format and
assigns them into the vLLM model state.

## Architecture

For MoE models like Qwen3-30B-A3B, the architecture (`Qwen3MoeForCausalLM`) has
no flax_nnx registration, so `get_model` falls through to `get_vllm_model`.
The state returned is:

```
model_runner.state = jax_view(shard_model_to_tpu(_VllmRunner(vllm_model)))
```

This is a plain `dict[str, jax.Array]` where values are the `._elem` of the
torchax.Tensor parameters.  The step function receives this dict as a regular
argument and calls `torch.func.functional_call(model, torch_view(state), ...)`.

## The DynamicJaxprTracer problem with direct dict assignment

Assigning a `jax.Array` produced from a different JAX mesh directly into the
state dict triggers a JIT-traced code path:

```python
# Broken — causes DynamicJaxprTracer at next inference call:
llm_state[key] = jax.device_put(weight, target_sharding)
```

The value is captured as a `DynamicJaxprTracer`. At inference time,
`torch.func.functional_call` tries to use it as a `torch.Tensor` and fails:

```
TypeError: Param(
  value=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>
               JitTracer(bfloat16[151936,2048])),
  ...
) is not an instance of torch.Tensor
```

### Old workaround (CPU roundtrip — slow)

```python
llm_state[key] = jax.device_put(np.asarray(weight), target_sharding)
```

`np.asarray()` materialises the value on host, so the subsequent `device_put`
takes a different (non-traced) constructor path.  This incurs a full TPU→CPU→TPU
roundtrip for every parameter, which is the dominant cost of weight sync.

## TPU-only solution (current implementation)

`model_runner.state` is a **plain `dict[str, jax.Array]`** produced by:
```
jax_view(shard_model_to_tpu(_VllmRunner(vllm_model)))
```
`jax_view` on a `dict[str, torchax.Tensor]` calls `.jax()` on each value,
yielding a plain dict of concrete `jax.Array`s. `dict.__setitem__` never
triggers JAX tracing.

Updating the state dict with a `jax.Array` produced from a different JAX mesh
is therefore safe — `jax.device_put` reshards on-device with no host copy:

```python
state = model_runner.state  # dict[str, jax.Array]
for key, weight in vllm_state.items():
    # On-device reshard only — no np.asarray() / host roundtrip.
    state[key] = jax.device_put(weight, state[key].sharding)
jax.effects_barrier()
```

Both the MaxText mesh and the vLLM mesh are backed by the same physical TPU
devices in this process, so `jax.device_put` resolves the cross-mesh copy
entirely on-device (all-to-all collective or no-op if layouts match).

### Why named_parameters() must NOT be used

`shard_model_to_tpu` builds the state dict by **extracting** params/buffers
from the module and converting them — it does *not* write the TPU-sharded
values back into the module.  Therefore `torch_module.named_parameters()`
still returns the **original** mix of CPU `torch.nn.parameter.Parameter`
objects and in-place-replaced `torchax.Tensor` objects, which is inconsistent
and cannot be safely fed to `jax_view` (which asserts that any `torch.Tensor`
must already be a `torchax.Tensor`).

The canonical source of truth for the vllm model runner weights is
`model_runner.state`.
