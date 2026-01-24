# MoE Sparse Matmul - JAX ragged_dot (`config.megablox=False`)

Source: `src/MaxText/layers/moe.py`
- `RoutedMoE.__call__`: lines 1930-1968
- `GateLogit`: lines 161-298
- `sparse_matmul`: lines 861-1358
- `gmm()` nested function: lines 875-970
- `permute()`: lines 590-641
- `unpermute()`: lines 643-727
- `local_permute()`: lines 729-766

## Overview

When `config.sparse_matmul=True` and `config.megablox=False` and `config.use_tokamax_gmm=False`, the MoE layer uses JAX's native `jax.lax.ragged_dot()` primitive for grouped matrix multiplication. This path has the same token-sorting logic as Megablox but uses a different kernel implementation.

**Trigger condition**: `config.sparse_matmul=True` AND `config.megablox=False` AND `config.use_tokamax_gmm=False`

**Key difference from Megablox**: Uses `jax.lax.ragged_dot()` instead of `mblx.gmm()`. XLA metadata hints are used to guide tiling optimization.

## Path Selection in gmm() Function

The `gmm()` nested function (lines 875-970) selects the implementation:

```
if config.use_tokamax_gmm:
    if config.quantization:
        → mblx.gmm(..., use_tokamax_backend=True)      # line 903
    else:
        → tokamax.ragged_dot(...)                       # line 916
else:
    if config.megablox:
        → mblx.gmm(...)                                 # line 926
    else:
        → jax.lax.ragged_dot(...)                       # line 952  ← THIS PATH
```

## Tensor Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              RoutedMoE.__call__                                  │
│                                                                                  │
│  INPUT                                                                           │
│  ─────                                                                           │
│  inputs: [B, S, M]                    (B=batch, S=seq, M=embed)                  │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                            GateLogit                                      │   │
│  │                                                                           │   │
│  │  inputs: [B, S, M]                                                        │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │         matmul                  │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  gate.kernel: [M, E]            │                                      │   │
│  │  │  sharding: ("embed", None)      │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  inputs @ kernel → output       │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  output: [B, S, E]                                                        │   │
│  │          │                                                                │   │
│  │          ▼  (optional)                                                    │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │  score_func (e.g. sigmoid)      │  (if config.routed_score_func)       │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼  (optional)                                                    │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │  + bias                         │  (if config.routed_bias)             │   │
│  │  │  gate.bias: [E]                 │                                      │   │
│  │  │  sharding: (None,)              │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  OUTPUTS: gate_logits: [B, S, E], pre_bias_logits: [B, S, E] or None      │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          │                                                                       │
│          │  Also extract MLP kernels from params:                                │
│          │  ┌─────────────────────────────────────────────────────────────┐      │
│          │  │  wi_0: [E, M, H]   sharding: wi_kernel_axes                 │      │
│          │  │  wi_1: [E, M, H]   sharding: wi_kernel_axes                 │      │
│          │  │  wo:   [E, H, M]   sharding: wo_kernel_axes                 │      │
│          │  └─────────────────────────────────────────────────────────────┘      │
│          │                                                                       │
│          ▼                                                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                         sparse_matmul (ragged_dot)                               │
│                                                                                  │
│  INPUTS                                                                          │
│  ──────                                                                          │
│  inputs: [B, S, M]                                                               │
│  gate_logits: [B, S, E]               (E=num_experts)                            │
│                                                                                  │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  inputs sharding (before shard_map)                          ║                │
│  ║  (batch_logical_axis, "activation_norm_length", None or      ║                │
│  ║   "activation_embed" if tensor_transpose_parallelism)        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                │
│                                                                                  │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  gate_logits sharding                                        ║                │
│  ║  (batch_logical_axis, "activation_norm_length", None)        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                           permute()                                     │     │
│  │                                                                         │     │
│  │  1. Reshape: [B, S, M] → [B*S, M]                                       │     │
│  │  2. get_topk: gate_logits → weights [B,S,K], selected_experts [B,S,K]   │     │
│  │  3. Replicate: inputs_2d repeated K times → [B*S*K, M]                  │     │
│  │  4. Sort by expert: argsort(flatten(selected_experts))                  │     │
│  │  5. Apply sort to get sorted_inputs                                     │     │
│  │  6. bincount to get group_sizes: [E]                                    │     │
│  │                                                                         │     │
│  │  IN:  inputs [B, S, M], gate_logits [B, S, E]                           │     │
│  │  OUT: sorted_inputs [T, M]      (T = B*S*K, sorted by expert)           │     │
│  │       sorted_selected_experts [T]  (indices to reverse sort)            │     │
│  │       weights [B, S, K]         (routing weights for combine)           │     │
│  │       group_sizes [E]           (count of tokens per expert)            │     │
│  │       sorted_experts [T]        (expert id for each sorted token)       │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          ▼  (if EP > 1, same as Megablox - see EP section below)                 │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │  [Ring-of-Experts / ragged_all_to_all / local_permute]                   │   │
│  │  (identical to Megablox path - handles token redistribution)             │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                       XLA Metadata Hints                                │     │
│  │                                                                         │     │
│  │  set_xla_metadata(                                                      │     │
│  │      ragged_dot_tiling=f"{tiling[0]},{tiling[1]},{tiling[2]}",          │     │
│  │      mosaic_fusion_group=f"{random_id}"                                 │     │
│  │  )                                                                      │     │
│  │                                                                         │     │
│  │  (Guides XLA compiler on tiling strategy for the ragged_dot op)         │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                       jax.lax.ragged_dot                                │     │
│  │                                                                         │     │
│  │  jax.lax.ragged_dot(                                                    │     │
│  │      lhs=sorted_inputs,       # [T, M]                                  │     │
│  │      rhs=w0,                  # [E, M, H]                               │     │
│  │      group_sizes=group_sizes, # [E]                                     │     │
│  │      preferred_element_type=dtype                                       │     │
│  │  )                                                                      │     │
│  │       │                                                                 │     │
│  │       ▼                                                                 │     │
│  │  layer_w0: [T, H]                                                       │     │
│  │                                                                         │     │
│  │  (group_sizes tells ragged_dot how many rows belong to each expert)     │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          │ (parallel)                                                            │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │  jax.lax.ragged_dot(sorted_inputs, w1, group_sizes, ...)                │     │
│  │                          ▼                                              │     │
│  │                   layer_w1: [T, H]                                      │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          │ (if tensor_transpose_parallelism)                                     │
│          ▼                                                                       │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │  jax.lax.psum(layer_w0, "tensor_transpose")                              │   │
│  │  jax.lax.psum(layer_w1, "tensor_transpose")                              │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────┐                                                           │
│  │activation (e.g.   │                                                           │
│  │silu(layer_w0))    │                                                           │
│  └───────────────────┘                                                           │
│          │                                                                       │
│          └──────────────┬───────────────────┘                                    │
│                         ▼                                                        │
│                 ┌───────────────────┐                                            │
│                 │  act(w0) * w1     │  (element-wise multiply)                   │
│                 └───────────────────┘                                            │
│                         │                                                        │
│                         ▼                                                        │
│                 intermediate_layer: [T, H]                                       │
│                         │                                                        │
│                         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │  jax.lax.ragged_dot(intermediate_layer, wo, group_sizes, ...)           │     │
│  │                                                                         │     │
│  │  intermediate_layer: [T, H]     wo: [E, H, M]                           │     │
│  │       └──────────────────┬────────────┘                                 │     │
│  │                          ▼                                              │     │
│  │                   output: [T, M']                                       │     │
│  │                   (M' = M / tensor_parallelism if TP > 1)               │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                         │                                                        │
│                         │ (if tensor_parallelism)                                │
│                         ▼                                                        │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │  jax.lax.psum_scatter(output, "tensor", scatter_dimension=1, tiled=True) │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│                         │                                                        │
│                         ▼  (if EP > 1)                                           │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │  [ragged_all_to_all / psum_scatter - reverse token redistribution]       │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│                         │                                                        │
│                         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                          unpermute()                                    │     │
│  │                                                                         │     │
│  │  1. Unsort: argsort(sorted_selected_experts) to reverse permutation     │     │
│  │  2. Reshape: [T, M] → [B*S, K, M]                                       │     │
│  │  3. Weight sum: einsum "BKE,BK -> BE" with routing weights              │     │
│  │  4. Reshape: [B*S, M] → [B, S, M]                                       │     │
│  │                                                                         │     │
│  │  IN:  intermediate [T, M], sorted_selected_experts [T], weights [B,S,K] │     │
│  │  OUT: output [B, S, M]                                                  │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                         │                                                        │
│                         ▼                                                        │
│  OUTPUT                                                                          │
│  ──────                                                                          │
│  output: [B, S, M]                                                               │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Dimension Key

| Symbol | Meaning | Typical Size |
|--------|---------|--------------|
| B | batch | varies |
| S | sequence length | varies |
| M | model/embed dim | e.g. 4096 |
| E | num_experts | e.g. 8, 64 |
| H | mlp/hidden dim | e.g. 14336 |
| K | num_experts_per_tok | e.g. 2 |
| T | total token-expert pairs | `B * S * K` |

## Sharding Summary

### Input/Output Sharding

| Component | Tensor | Type | Shape | Sharding |
|-----------|--------|------|-------|----------|
| GateLogit | kernel | weight | `[M, E]` | `("embed", None)` |
| GateLogit | bias | weight | `[E]` | `(None,)` |
| sparse_matmul | inputs | act | `[B, S, M]` | `(batch_axis, "activation_norm_length", None or "activation_embed")` |
| sparse_matmul | gate_logits | act | `[B, S, E]` | `(batch_axis, "activation_norm_length", None)` |
| sparse_matmul | output | act | `[B, S, M]` | `(batch_axis, "activation_norm_length", "activation_embed")` |

### Weight Kernel Sharding (inside shard_map)

Identical to Megablox path. The kernel sharding depends on configuration (lines 1020-1047):

| Config | w0/w1 pspec | wo pspec |
|--------|-------------|----------|
| `shard_exp_on_fsdp` + fixed quantization | `wi_kernel_axes` | `wo_kernel_axes` |
| `shard_exp_on_fsdp` | `("embed_tensor_transpose", None, "mlp_no_fsdp")` | `("embed_tensor_transpose", "mlp_no_fsdp", None)` |
| `use_2d_fsdp_sharding` | `("embed_tensor_transpose", "mlp_no_fsdp", None)` | same |
| Default | `("exp", "embed_tensor_transpose", "mlp_no_fsdp")` | `("exp", "mlp_no_fsdp", "embed_tensor_transpose")` |

### Key Difference: No weight_gather_axes

Unlike Megablox, `jax.lax.ragged_dot()` does NOT have a `weight_gather_axes` parameter. Weight gathering must be handled explicitly if needed, or the kernel must assume weights are fully replicated/gathered before the call.

## Comparison: ragged_dot vs Megablox

| Aspect | jax.lax.ragged_dot | mblx.gmm (Megablox) |
|--------|-------------------|---------------------|
| Implementation | JAX native primitive | Custom Pallas kernel |
| Tiling control | XLA metadata hints | Explicit tiling params |
| Weight gathering | Not supported | `weight_gather_axes` param |
| Quantization | Via separate QTensor handling | Built-in support |
| Code path | lines 938-967 | lines 925-937 |

### XLA Metadata Hints (lines 948-951)

```python
with set_xla_metadata(
    ragged_dot_tiling=f"{tiling[0]},{tiling[1]},{tiling[2]}",
    mosaic_fusion_group=f"{random_id}"
):
    output = jax.lax.ragged_dot(...)
```

These hints guide XLA's compilation:
- `ragged_dot_tiling`: Suggests tile sizes for the operation
- `mosaic_fusion_group`: Groups ops for potential fusion

### Quantization Handling (lines 940-967)

When the kernel is a `QTensor` (quantized):
1. Extract `rhs_inputs = kernel.qvalue` (the quantized values)
2. Call `ragged_dot` with quantized values
3. Multiply output by scale: `output *= scales[expert_assignments]`

## Permute/Unpermute Example

(Identical to Megablox - see moe_sparse_megablox.md for detailed walkthrough)

Consider: 4 tokens, 4 experts, top-k=2

```
Input:  T0→[E1,E2], T1→[E1,E3], T2→[E0,E1], T3→[E2,E3]

After permute():
  sorted_inputs: [T2, T0, T1, T2, T0, T3, T1, T3]  (sorted by expert)
  group_sizes:   [1, 3, 2, 2]  (E0:1, E1:3, E2:2, E3:2)

ragged_dot processes:
  Expert 0: rows 0:1
  Expert 1: rows 1:4
  Expert 2: rows 4:6
  Expert 3: rows 6:8

After unpermute():
  output[T0] = 0.6 * E1(T0) + 0.4 * E2(T0)
  output[T1] = 0.7 * E1(T1) + 0.3 * E3(T1)
  output[T2] = 0.5 * E0(T2) + 0.5 * E1(T2)
  output[T3] = 0.8 * E2(T3) + 0.2 * E3(T3)
```

---

## Differences between EP and no-EP

The EP handling is **identical to the Megablox path**. Both paths share the same:
- `permute()` / `unpermute()` functions
- `local_permute()` for intra-shard sorting
- `ragged_all_to_all()` for token redistribution
- `psum_scatter()` for ring-of-experts aggregation

### When EP = 1 (no expert parallelism)

Simple flow:
1. `permute()` → sorts tokens by expert
2. `ragged_dot()` × 3 (w0, w1, wo) → process all experts locally
3. `unpermute()` → restore order and apply weights

### When EP > 1 (expert parallelism enabled)

Three modes (same as Megablox):

#### Mode A: Ring-of-Experts
1. `all_gather` inputs across all EP shards
2. `permute()` with `roll_to_expert_id`
3. Each shard processes its subset of experts
4. `unpermute()` locally
5. `psum_scatter` to aggregate

#### Mode B: Batch Sharded by Expert
1. `permute()` locally
2. `all_gather` group_sizes
3. `ragged_all_to_all()` → redistribute tokens TO expert shards
4. `local_permute()` → re-sort within shard
5. `ragged_dot()` × 3
6. Local unpermute
7. `ragged_all_to_all()` → redistribute BACK to batch shards
8. `unpermute()`

#### Mode C: Batch NOT Sharded by Expert
1. `permute()` locally
2. `local_permute(is_offset=True)`
3. `ragged_dot()` × 3
4. `ragged_all_to_all()` → redistribute outputs
5. `unpermute()`

---

## Formal Specification (Machine-Readable)

### Definitions

```
INPUTS:
  B = batch size
  S = sequence length (number of tokens)
  E = number of experts
  K = num_experts_per_tok (top-k)
  M = model/embedding dimension
  H = MLP hidden dimension

DERIVED:
  T = B × S × K  (total token-expert assignment pairs)
```

### jax.lax.ragged_dot Semantics

```
jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type):
  - lhs: [T, K_in]  (sorted input tokens)
  - rhs: [E, K_in, K_out]  (expert weights, one matrix per expert)
  - group_sizes: [E]  (count of tokens per expert)
  - preferred_element_type: output dtype

  Semantics:
    output = zeros[T, K_out]
    offset = 0
    for e in 0..E:
      n = group_sizes[e]
      output[offset:offset+n] = lhs[offset:offset+n] @ rhs[e]
      offset += n
    return output

  Invariant:
    sum(group_sizes) == T
```

### Full Forward Pass (Pseudocode)

```python
def sparse_matmul_ragged_dot(inputs, gate_logits, w0, w1, wo):
    # inputs: [B, S, M], gate_logits: [B, S, E]
    # w0, w1: [E, M, H], wo: [E, H, M]

    # Step 1: Permute tokens by expert assignment
    sorted_inputs, sort_indices, weights, group_sizes = permute(inputs, gate_logits)
    # sorted_inputs: [T, M], T = B*S*K

    # Step 2: Up-projection with ragged_dot
    with set_xla_metadata(ragged_dot_tiling=..., mosaic_fusion_group=...):
        layer_w0 = jax.lax.ragged_dot(sorted_inputs, w0, group_sizes)  # [T, H]
        layer_w1 = jax.lax.ragged_dot(sorted_inputs, w1, group_sizes)  # [T, H]

    # Step 3: Activation and gating
    intermediate = activation(layer_w0) * layer_w1  # [T, H]

    # Step 4: Down-projection with ragged_dot
    with set_xla_metadata(ragged_dot_tiling=..., mosaic_fusion_group=...):
        output = jax.lax.ragged_dot(intermediate, wo, group_sizes)  # [T, M]

    # Step 5: Unpermute and combine
    output = unpermute(output, sort_indices, weights)  # [B, S, M]

    return output
```

### Invariants

```
1. ragged_dot preserves token count:
   output.shape[0] == lhs.shape[0] == sum(group_sizes)

2. Each expert processes its assigned tokens:
   for e in 0..E:
     start = sum(group_sizes[:e])
     end = sum(group_sizes[:e+1])
     output[start:end] = lhs[start:end] @ rhs[e]

3. Group boundaries are contiguous and non-overlapping:
   groups partition [0, T) into E consecutive segments

4. Weight matrix broadcasting:
   Each row of lhs in group e is multiplied by the same rhs[e]
```

### Tiling Hints

The tiling parameter passed to XLA metadata has three components:
```
tiling = (batch_seq_tile, embed_tile, mlp_tile)

wi_tile_size uses:
  - config.wi_tile_fwd_batch_seq
  - config.wi_tile_fwd_embed_dim
  - config.wi_tile_fwd_mlp_dim

wo_tile_size uses:
  - config.wo_tile_fwd_batch_seq
  - config.wo_tile_fwd_embed_dim
  - config.wo_tile_fwd_mlp_dim
```

For ragged_dot (non-megablox), tiling is clamped to actual dimensions (lines 896-900):
```python
tiling = (
    min(tiling[0], m),  # m = num_tokens
    min(tiling[1], k),  # k = input_dim
    min(tiling[2], n),  # n = output_dim
)
```
