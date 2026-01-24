# MoE Sparse Matmul - Megablox (`config.megablox=True`)

Source: `src/MaxText/layers/moe.py`
- `RoutedMoE.__call__`: lines 1930-1968
- `GateLogit`: lines 161-298
- `sparse_matmul`: lines 861-1358
- `gmm()` nested function: lines 875-970
- `permute()`: lines 590-641
- `unpermute()`: lines 643-727
- `local_permute()`: lines 729-766

## Overview

When `config.sparse_matmul=True` and `config.megablox=True`, the MoE layer uses the Megablox grouped matrix multiplication kernel. Tokens are sorted by expert assignment, processed in groups, and then unsorted back to original order.

**Trigger condition**: `config.sparse_matmul=True` AND `config.megablox=True`

**Key difference from dense path**: Instead of computing all token-expert pairs and masking, sparse matmul sorts tokens by expert and uses grouped matmul to process only the assigned token-expert pairs efficiently.

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
│                          sparse_matmul (Megablox)                                │
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
│          ▼  (if EP > 1 and ring_of_experts)                                      │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │                  Ring-of-Experts Strategy                                │   │
│  │                                                                          │   │
│  │  1. all_gather inputs across EP shards                                   │   │
│  │  2. Each shard processes subset of experts                               │   │
│  │  3. psum_scatter to aggregate outputs                                    │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│          │                                                                       │
│          ▼  (if EP > 1 and batch_sharded_by_expert)                              │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │                ragged_all_to_all (batch sharded)                         │   │
│  │                                                                          │   │
│  │  Redistribute tokens from batch shards to expert shards                  │   │
│  │  Each expert shard receives tokens destined for its experts              │   │
│  │                                                                          │   │
│  │  jax.lax.ragged_all_to_all(sorted_inputs, ..., axis_name="expert")       │   │
│  │       │                                                                  │   │
│  │       ▼                                                                  │   │
│  │  local_permute() → re-sort tokens within shard by local expert id        │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│          │                                                                       │
│          ▼  (if EP > 1 and NOT batch_sharded_by_expert)                          │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │                local_permute (batch replicated)                          │   │
│  │                                                                          │   │
│  │  Batch is replicated across EP shards                                    │   │
│  │  local_permute(is_offset=True) selects tokens for local experts          │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                         Megablox GMM                                    │     │
│  │                                                                         │     │
│  │  mblx.gmm(lhs=sorted_inputs, rhs=w0, group_sizes=group_sizes,           │     │
│  │           tiling=wi_tile_size, weight_gather_axes=wi_gather_axes)       │     │
│  │                                                                         │     │
│  │  sorted_inputs: [T, M]          w0: [E, M, H]                           │     │
│  │       └──────────────────┬────────────┘                                 │     │
│  │                          ▼                                              │     │
│  │                   layer_w0: [T, H]                                      │     │
│  │                                                                         │     │
│  │  (group_sizes tells gmm how many tokens belong to each expert)          │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          │ (parallel)                                                            │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │  mblx.gmm(sorted_inputs, w1, group_sizes, wi_tile_size, wi_gather_axes) │     │
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
│  │  mblx.gmm(intermediate_layer, wo, group_sizes, wo_tile_size,            │     │
│  │           wo_gather_axes)                                               │     │
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
│                         ▼  (if EP > 1 and batch_sharded_by_expert)               │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │                ragged_all_to_all (reverse)                               │   │
│  │                                                                          │   │
│  │  1. local unpermute via argsort(local_sorted_indices)                    │   │
│  │  2. ragged_all_to_all to redistribute outputs back to batch shards       │   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │
│                         │                                                        │
│                         ▼  (if EP > 1 and NOT batch_sharded_by_expert)           │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │
│  │                ragged_all_to_all (batch replicated)                      │   │
│  │                                                                          │   │
│  │  Single ragged_all_to_all to redistribute outputs                        │   │
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

The kernel sharding depends on configuration (lines 1020-1047):

| Config | w0/w1 pspec | wo pspec |
|--------|-------------|----------|
| `shard_exp_on_fsdp` + fixed quantization | `wi_kernel_axes` | `wo_kernel_axes` |
| `shard_exp_on_fsdp` | `("embed_tensor_transpose", None, "mlp_no_fsdp")` | `("embed_tensor_transpose", "mlp_no_fsdp", None)` |
| `use_2d_fsdp_sharding` | `("embed_tensor_transpose", "mlp_no_fsdp", None)` | same |
| Default | `("exp", "embed_tensor_transpose", "mlp_no_fsdp")` | `("exp", "mlp_no_fsdp", "embed_tensor_transpose")` |

### Weight Gathering

When `weight_gather=True` (shard_exp_on_fsdp + fixed quantization), Megablox performs implicit weight gathering via the `weight_gather_axes` parameter:
- `wi_gather_axes`: gathers expert dim (0) and hidden dim (2) if sharded
- `wo_gather_axes`: gathers expert dim (0) and hidden dim (1) if sharded

## Permute/Unpermute Example

Consider a simple example with:
- **4 tokens** (S=4, B=1)
- **4 experts** (E=4)
- **top-k = 2** (each token routed to 2 experts)

### Step 1: Router Assigns Experts

`get_topk` returns:

| Token | selected_experts | weights (softmax) |
|-------|------------------|-------------------|
| T0    | [1, 2]           | [0.6, 0.4]        |
| T1    | [1, 3]           | [0.7, 0.3]        |
| T2    | [0, 1]           | [0.5, 0.5]        |
| T3    | [2, 3]           | [0.8, 0.2]        |

### Step 2: Flatten and Replicate

```
flatten(selected_experts) = [1, 2, 1, 3, 0, 1, 2, 3]
                             T0k0 T0k1 T1k0 T1k1 T2k0 T2k1 T3k0 T3k1

replicated_inputs = [T0, T0, T1, T1, T2, T2, T3, T3]  (each token repeated K=2 times)
```

### Step 3: Sort by Expert

```
argsort([1, 2, 1, 3, 0, 1, 2, 3]) = [4, 0, 2, 5, 1, 6, 3, 7]
                                    E0  E1 E1 E1  E2 E2  E3 E3

sorted_inputs = [T2, T0, T1, T2, T0, T3, T1, T3]  (grouped by expert)
group_sizes = [1, 3, 2, 2]  (E0 has 1, E1 has 3, E2 has 2, E3 has 2)
```

### Step 4: Grouped Matmul (Megablox)

```
Expert 0 processes: [T2]           (rows 0:1)
Expert 1 processes: [T0, T1, T2]   (rows 1:4)
Expert 2 processes: [T0, T3]       (rows 4:6)
Expert 3 processes: [T1, T3]       (rows 6:8)
```

Megablox `gmm()` handles this grouping internally using `group_sizes`.

### Step 5: Unpermute and Weight Sum

```
1. Reverse sort: outputs[argsort([4, 0, 2, 5, 1, 6, 3, 7])]
   → [E1(T0), E2(T0), E1(T1), E3(T1), E0(T2), E1(T2), E2(T3), E3(T3)]

2. Reshape to [B*S, K, M]:
   → [[E1(T0), E2(T0)], [E1(T1), E3(T1)], [E0(T2), E1(T2)], [E2(T3), E3(T3)]]

3. Weight sum einsum "BKE, BK -> BE":
   output[T0] = 0.6 * E1(T0) + 0.4 * E2(T0)
   output[T1] = 0.7 * E1(T1) + 0.3 * E3(T1)
   output[T2] = 0.5 * E0(T2) + 0.5 * E1(T2)
   output[T3] = 0.8 * E2(T3) + 0.2 * E3(T3)
```

### Key Insight

Unlike dense matmul, **no computation is wasted on unassigned token-expert pairs**. The grouped matmul processes exactly the assigned pairs, making it more efficient when `K << E`.

---

## Differences between EP and no-EP

### When EP = 1 (no expert parallelism)

Simple flow:
1. `permute()` → sorts tokens by expert
2. `gmm()` × 3 (w0, w1, wo) → process all experts locally
3. `unpermute()` → restore order and apply weights

### When EP > 1 (expert parallelism enabled)

Three modes are supported:

#### Mode A: Ring-of-Experts (`config.use_ring_of_experts=True`)

1. `all_gather` inputs across all EP shards (line 1085)
2. `permute()` with `roll_to_expert_id` to route to local experts
3. Each shard processes its subset of experts
4. `unpermute()` locally
5. `psum_scatter` to aggregate partial outputs across shards (line 1255)

#### Mode B: Batch Sharded by Expert (`is_batch_sharded_by_expert=True`)

1. `permute()` locally on each batch shard
2. `all_gather` group_sizes across EP shards (line 1116)
3. `ragged_all_to_all()` to redistribute tokens TO expert shards (line 1135)
4. `local_permute()` to re-sort within each shard (line 1145)
5. `gmm()` × 3 → process local experts
6. Local unpermute via `argsort(local_sorted_indices)` (line 1271)
7. `ragged_all_to_all()` to redistribute outputs BACK to batch shards (line 1281)
8. `unpermute()` to restore final order

#### Mode C: Batch NOT Sharded by Expert (`is_batch_sharded_by_expert=False`)

1. `permute()` locally (batch is replicated across EP shards)
2. `local_permute(is_offset=True)` to select tokens for local experts (line 1153)
3. `gmm()` × 3 → process local experts
4. Single `ragged_all_to_all()` to redistribute outputs (line 1301)
5. `unpermute()` to restore final order

### Key Difference from Dense Path

- **Dense path**: Explicitly calls `maybe_all_gather_kernel_weight_in_expert_parallelism()` to all-gather weights before computation
- **Megablox sparse path**: Weight gathering is implicit via `weight_gather_axes` parameter passed to `mblx.gmm()`; tokens are redistributed instead of weights

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

### Data Structures

```
selected_experts: int[B, S, K]
  - For each token, the K expert indices it is routed to
  - Values in range [0, E)

weights: float[B, S, K]
  - Softmax routing weights for each token's K expert assignments
  - Sum over K equals 1.0 for each token

sorted_selected_experts: int[T]
  - Indices that sort flatten(selected_experts)
  - Used to reverse the sort in unpermute()

group_sizes: int[E]
  - Count of tokens assigned to each expert
  - group_sizes[e] = number of tokens routed to expert e
  - sum(group_sizes) = T

sorted_inputs: float[T, M]
  - Tokens sorted by expert assignment
  - Rows 0:group_sizes[0] are for expert 0
  - Rows group_sizes[0]:group_sizes[0]+group_sizes[1] are for expert 1
  - etc.
```

### Permute Algorithm

```python
def permute(inputs, gate_logits):
    # inputs: [B, S, M], gate_logits: [B, S, E]

    # Step 1: Flatten batch and sequence
    inputs_2d = reshape(inputs, [B*S, M])

    # Step 2: Get top-k experts per token
    weights, selected_experts = get_topk(gate_logits)
    # weights: [B, S, K], selected_experts: [B, S, K]

    # Step 3: Flatten expert assignments
    flatten_experts = ravel(selected_experts)  # [T]

    # Step 4: Get sort order by expert
    sorted_indices = argsort(flatten_experts)  # [T]

    # Step 5: Replicate inputs K times and sort
    replicated = repeat(inputs_2d, K, axis=0)  # [T, M]
    sorted_inputs = replicated[sorted_indices]  # [T, M]

    # Step 6: Count tokens per expert
    group_sizes = bincount(flatten_experts, length=E)  # [E]

    return sorted_inputs, sorted_indices, weights, group_sizes
```

### Megablox GMM Semantics

```
mblx.gmm(lhs, rhs, group_sizes, weight_gather_axes):
  - lhs: [T, K_in]  (sorted input tokens)
  - rhs: [E, K_in, K_out]  (expert weights)
  - group_sizes: [E]  (count per expert)
  - weight_gather_axes: list of (axis_name, dim) for implicit weight gathering

  Semantics:
    output = zeros[T, K_out]
    offset = 0
    for e in 0..E:
      n = group_sizes[e]
      output[offset:offset+n] = lhs[offset:offset+n] @ rhs[e]
      offset += n
    return output
```

### Unpermute Algorithm

```python
def unpermute(intermediate, sorted_indices, weights, B, S):
    # intermediate: [T, M], sorted_indices: [T], weights: [B, S, K]

    # Step 1: Reverse the sort
    unsorted = intermediate[argsort(sorted_indices)]  # [T, M]

    # Step 2: Reshape to [B*S, K, M]
    reshaped = reshape(unsorted, [B*S, K, M])

    # Step 3: Weight sum
    weights_flat = reshape(weights, [B*S, K])
    output = einsum("BKM,BK->BM", reshaped, weights_flat)  # [B*S, M]

    # Step 4: Reshape to [B, S, M]
    return reshape(output, [B, S, M])
```

### Invariants

```
1. Every token appears exactly K times in sorted_inputs:
   for all s in 0..B*S: count of s in sorted_inputs == K

2. Group sizes sum to total token-expert pairs:
   sum(group_sizes) == T == B * S * K

3. Tokens within each expert group are contiguous:
   for all e: rows [cumsum(group_sizes[:e]) : cumsum(group_sizes[:e+1])]
             all belong to expert e

4. Unpermute perfectly reverses permute:
   unpermute(gmm_output, sorted_indices, weights) reconstructs
   the original token ordering with weighted expert outputs
```
