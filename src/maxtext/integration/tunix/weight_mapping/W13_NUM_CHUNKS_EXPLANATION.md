# Origin of `num_chunks=4` in `_to_mlp_expert_gate_up`

## Summary

The magic number `num_chunks=4` in the MoE gate+up weight interleaving is **not** a model-specific constant. It equals the **`MLP_TENSOR` mesh-axis product of the vLLM torchax mesh**, which in the default TP=4 configuration happens to be 4. It must always match the vLLM-side `w13_reorder_size`.

---

## Call Chain: How vLLM Pre-processes `w13_weight`

When `tpu_inference` loads an unquantized MoE model for the GMM_TP backend, it runs the following pipeline:

### Step 1 — Derive `w13_reorder_size` from the mesh

```python
# tpu_inference/layers/vllm/quantization/unquantized.py
w13_reorder_size = get_mesh_shape_product(self.mesh, ShardingAxisName.MLP_TENSOR)
process_moe_weights(..., moe_backend=GMM_TP, w13_reorder_size=w13_reorder_size, ...)
```

`ShardingAxisName.MLP_TENSOR` resolves to `'model'` (the tensor-parallel axis) in the default 2D sharding case.

### Step 2 — Reorder `[gate | up]` into per-shard chunks

```python
# tpu_inference/layers/common/process_weights/moe_weights.py  (GMM_TP branch)
w13_weight = jnp.swapaxes(w13_weight, 1, 2)          # (E, 2*inter, hidden) → (E, hidden, 2*inter)
w13_weight = reorder_concatenated_tensor_for_sharding(
    w13_weight,
    output_sizes=[intermediate_size, intermediate_size],
    n_shards=w13_reorder_size,
    dim=2,
)
```

`reorder_concatenated_tensor_for_sharding` transforms the layout from:
```
dim-2: [gate_0 ... gate_N | up_0 ... up_N]
```
into:
```
dim-2: [g₀ u₀ | g₁ u₁ | g₂ u₂ | g₃ u₃]  (n_shards=4 interleaved pairs)
```

### Step 3 — Shard `dim-2` across `MLP_TENSOR`

```python
# shard_moe_weights, GMM_TP branch
w13_weight = NamedSharding(mesh, P(None, None, ShardingAxisName.MLP_TENSOR))
# shape: (num_experts, hidden_dim, 2*intermediate_size)
```

Each TP rank receives one contiguous `[gate_chunk_i, up_chunk_i]` pair, enabling efficient local matmuls without cross-device communication.

---

## Why `num_chunks` Must Equal `w13_reorder_size`

After `shard_moe_weights`, each TP device holds:
```
(E, hidden, 2*inter / n_shards)  ← contains both gate and up slices for its TP rank
```

For vLLM's GMM kernel to correctly separate gate from up locally, the chunks **must already be paired** in the pre-sharded layout. MaxText's conversion must produce this same interleaved layout so that when vLLM re-shards it (during the weight sync), it arrives on each device in the expected format.

**`num_chunks` in the MaxText converter = `w13_reorder_size` in vLLM = `MLP_TENSOR` mesh axis product.**

---

## Why TP=4 Gives `num_chunks=4`

With `LLM(..., tensor_parallel_size=4)` and no `enable_dp_attention`:

| Mesh axis | Size |
|-----------|------|
| `data`    | 1    |
| `attn_dp` | 1    |
| `attn_dp_expert` | 1 |
| `expert`  | 1    |
| `model`   | **4** |

```python
get_mesh_shape_product(mesh, 'model')  # → 4
```

So `w13_reorder_size = 4`, and `num_chunks` must be `4`.

---

## Why TP=2/DP=2 with `num_chunks=2` Produces Garbage

Changing `num_chunks=2` for the conversion is **correct** for the vLLM TP=2 side (`model=2` → `w13_reorder_size=2`). The garbage comes from the MaxText loading config being hardcoded to `ici_expert_parallelism=4` regardless of the vLLM config:

```python
config_ref = pyconfig.initialize(
    ...
    ici_expert_parallelism=4,   # ← always 4, doesn't track vLLM TP
    ...
)
```

MaxText EP=4 shards the expert dimension across 4 chips. When you then reshard to a vLLM mesh of `(DP=2, model=2)`, the expert-partitioned layout from MaxText and the tensor-parallel layout expected by vLLM do not match cleanly. The fix requires aligning the MaxText parallelism config with the vLLM mesh topology **and** setting `num_chunks=2`.

---

## Making `num_chunks` Non-Magic

Instead of hardcoding, derive it from the vLLM state after the `LLM` object is initialized:

```python
# After:  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
_sample_w13 = next(v for k, v in llm_state.items() if 'w13_weight' in k)
# w13 sharding spec is P(None, None, MLP_TENSOR); dim 2 is the TP-sharded dim.
mlp_tensor_size = _sample_w13.sharding.mesh.shape.get('model', 1)

converter = MaxTextToVLLMConverter(config, mesh, num_tp_chunks=mlp_tensor_size)
```

Or simply parameterize both `LLM(tensor_parallel_size=N)` and the converter with the same `N`, keeping them in sync explicitly.
