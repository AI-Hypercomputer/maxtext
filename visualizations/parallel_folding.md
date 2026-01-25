# MoE Parallel Folding Implementation

Source: `~/moe_parallel_folding/src/model/`
- `moe.py`: Router, StackedExperts, MoEFFN (dense einsum path)
- `moe_dispatcher.py`: pack_tokens, combine_outputs, sharded dispatch/combine
- `config.py`: MeshConfig, ModelConfig

## Overview

This implementation provides two MoE execution paths:

1. **Dense Einsum Path** (`MoEFFN` in moe.py): Uses dispatch/combine masks with einsum operations. Similar to MaxText's dense_matmul with token dropping. **(Active in model.py)**

2. **Parallel Folding Path** (`moe_dispatcher.py`): Uses explicit token packing and `all_to_all` collectives to redistribute tokens to expert-owning devices. **(Test/experimental only)**

The key innovation is the **parallel folding** approach: tokens are packed into expert-indexed buffers, redistributed via `all_to_all`, processed by local experts, and results are returned via another `all_to_all`.

## Mesh Configuration

Two mesh configurations are used (same physical devices, different logical views):

| Mesh | Shape | Axes | Purpose |
|------|-------|------|---------|
| Attention | `(2, 2, 2)` | `('dp', 'cp', 'tp')` | Data/Context/Tensor parallelism for attention |
| MoE | `(8, 1)` | `('ep', 'etp')` | Expert parallelism for MoE layers |

Token sharding:
- Attention: tokens sharded over `('dp', 'cp')` = 4-way
- MoE: tokens sharded over `('ep',)` = 8-way

---

## Path 1: Dense Einsum (MoEFFN)

This path uses dispatch/combine masks, similar to MaxText's dense_matmul with token dropping.

### Tensor Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 MoEFFN.__call__                                  │
│                                                                                  │
│  INPUT                                                                           │
│  ─────                                                                           │
│  x: [B, S, M]                       (B=batch, S=seq, M=d_model)                  │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                              Router                                       │   │
│  │                                                                           │   │
│  │  x: [B, S, M]                                                             │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │         matmul + bias           │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  gate_kernel: [M, E]            │                                      │   │
│  │  │  sharding: (None, None)         │  ← replicated                        │   │
│  │  │  gate_bias: [E]                 │                                      │   │
│  │  │  sharding: (None,)              │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  logits = x @ kernel + bias     │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  logits: [B, S, E]                                                        │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │         top_k                   │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  jax.lax.top_k(logits, K)       │                                      │   │
│  │  │  → top_k_logits: [B, S, K]      │                                      │   │
│  │  │  → top_k_indices: [B, S, K]     │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │     softmax(top_k_logits)       │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  weights: [B, S, K]             │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  OUTPUTS: top_k_indices, weights, logits                                  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                      Router.generate_masks                                │   │
│  │                                                                           │   │
│  │  Capacity calculation:                                                    │   │
│  │  capacity = ceil((B * S * K) / E * capacity_factor)                       │   │
│  │                                                                           │   │
│  │  1. One-hot encode expert assignments: [B, S, K] → [B, S, K, E]           │   │
│  │  2. Sequential scan to assign capacity positions per expert               │   │
│  │  3. Create dispatch_mask: [B, S, E, C] - bool, position < capacity        │   │
│  │  4. Create combine_mask: [B, S, E, C] - float, routing weights            │   │
│  │                                                                           │   │
│  │  OUTPUTS: dispatch_mask, combine_mask, capacity                           │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                            DISPATCH                                     │     │
│  │                                                                         │     │
│  │  einsum: 'BSM, BSEC -> EBCM'                                            │     │
│  │                                                                         │     │
│  │  x: [B, S, M]              dispatch_mask: [B, S, E, C]                  │     │
│  │                  ↘            ↙                                         │     │
│  │                    expert_inputs: [E, B, C, M]                          │     │
│  │                                                                         │     │
│  │  (tokens gathered per expert, up to capacity C)                         │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          ▼                                                                       │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  expert_inputs sharding constraint                           ║                │
│  ║  P('ep', None, None, None)                                   ║                │
│  ║  (E dimension sharded over 'ep' axis)                        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                         StackedExperts                                    │   │
│  │                                                                           │   │
│  │  expert_inputs: [E, B, C, M]                                              │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │           MLP Up (w1)           │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  einsum: 'EBCM, EMH -> EBCH'    │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  w1_kernel: [E, M, H]           │                                      │   │
│  │  │  sharding: ('ep', None, None)   │                                      │   │
│  │  │  w1_bias: [E, H]                │                                      │   │
│  │  │  sharding: ('ep', None)         │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  hidden: [E, B, C, H]                                                     │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │        GELU activation          │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  ┌─────────────────────────────────┐                                      │   │
│  │  │          MLP Down (w2)          │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  einsum: 'EBCH, EHM -> EBCM'    │                                      │   │
│  │  │                                 │                                      │   │
│  │  │  w2_kernel: [E, H, M]           │                                      │   │
│  │  │  sharding: ('ep', None, None)   │                                      │   │
│  │  │  w2_bias: [E, M]                │                                      │   │
│  │  │  sharding: ('ep', None)         │                                      │   │
│  │  └─────────────────────────────────┘                                      │   │
│  │          │                                                                │   │
│  │          ▼                                                                │   │
│  │  expert_outputs: [E, B, C, M]                                             │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                            COMBINE                                      │     │
│  │                                                                         │     │
│  │  einsum: 'EBCM, BSEC -> BSM'                                            │     │
│  │                                                                         │     │
│  │  expert_outputs: [E, B, C, M]    combine_mask: [B, S, E, C]             │     │
│  │                    ↘                ↙                                   │     │
│  │                      output: [B, S, M]                                  │     │
│  │                                                                         │     │
│  │  (expert outputs scattered back to token positions, weighted)           │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│          ▼                                                                       │
│  OUTPUT                                                                          │
│  ──────                                                                          │
│  output: [B, S, M]                                                               │
│  metrics: {tokens_per_expert, dropped_per_expert, aux_loss}                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Sharding Summary (Dense Path)

| Component | Tensor | Type | Shape | Sharding |
|-----------|--------|------|-------|----------|
| Router | gate_kernel | weight | `[M, E]` | `(None, None)` - replicated |
| Router | gate_bias | weight | `[E]` | `(None,)` - replicated |
| StackedExperts | w1_kernel | weight | `[E, M, H]` | `('ep', None, None)` |
| StackedExperts | w1_bias | weight | `[E, H]` | `('ep', None)` |
| StackedExperts | w2_kernel | weight | `[E, H, M]` | `('ep', None, None)` |
| StackedExperts | w2_bias | weight | `[E, M]` | `('ep', None)` |
| MoEFFN | expert_inputs | act | `[E, B, C, M]` | `P('ep', None, None, None)` |

---

## Path 2: Parallel Folding (moe_dispatcher.py)

This path uses explicit token packing and `all_to_all` collectives for expert parallelism.

### Tensor Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    create_sharded_dispatch_combine (kernel)                      │
│                                                                                  │
│  Each device processes its local shard of tokens                                 │
│                                                                                  │
│  INPUT (per device)                                                              │
│  ─────                                                                           │
│  x_shard: [tokens_local, M]         (tokens_local = total_tokens / num_devices)  │
│  top_k_ids_shard: [tokens_local, K]                                              │
│  top_k_weights_shard: [tokens_local, K]                                          │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                           pack_tokens                                     │   │
│  │                                                                           │   │
│  │  Organizes tokens into expert-indexed send buffer                         │   │
│  │                                                                           │   │
│  │  1. Initialize:                                                           │   │
│  │     sendbuf: [E, C, M] = zeros                                            │   │
│  │     token_indices: [E, C] = -1 (padding marker)                           │   │
│  │     top_k_slot: [E, C] = -1                                               │   │
│  │     weights: [E, C] = 0                                                   │   │
│  │     expert_counts: [E] = 0                                                │   │
│  │                                                                           │   │
│  │  2. Sequential scan over all (token, k) assignments:                      │   │
│  │     for each (token_idx, k_idx, expert_id, weight):                       │   │
│  │       pos = expert_counts[expert_id]                                      │   │
│  │       if pos < capacity:                                                  │   │
│  │         sendbuf[expert_id, pos] = x[token_idx]                            │   │
│  │         token_indices[expert_id, pos] = token_idx                         │   │
│  │         weights[expert_id, pos] = weight                                  │   │
│  │         expert_counts[expert_id] += 1                                     │   │
│  │       else:                                                               │   │
│  │         # token dropped for this expert                                   │   │
│  │                                                                           │   │
│  │  OUTPUT:                                                                  │   │
│  │    sendbuf: [E, C, M]                                                     │   │
│  │    metadata: DispatchMetadata(token_indices, top_k_slot, weights, dropped)│   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          │  sendbuf: [E, C, M] - tokens organized by destination expert          │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                      all_to_all (forward)                                 │   │
│  │                                                                           │   │
│  │  lax.all_to_all(sendbuf, axis_name='ep', split_axis=0, concat_axis=0)     │   │
│  │                                                                           │   │
│  │  Before: Each device has [E, C, M] with tokens FOR each expert            │   │
│  │  After:  Each device has [E, C, M] with tokens FROM each device           │   │
│  │                                                                           │   │
│  │  Device 0 (expert 0):                                                     │   │
│  │    receives tokens destined for expert 0 from ALL devices                 │   │
│  │                                                                           │   │
│  │  Also all_to_all for metadata: token_indices, weights, top_k_slot         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          │  recvbuf: [E, C, M] - tokens from all devices for local expert        │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                        Expert Computation                                 │   │
│  │                                                                           │   │
│  │  expert_id = lax.axis_index('ep')  ← which expert this device owns        │   │
│  │                                                                           │   │
│  │  expert_input = recvbuf.reshape(-1, M)   # [E*C, M]                       │   │
│  │  expert_output = expert_fn(expert_id, expert_input)                       │   │
│  │  expert_output = expert_output.reshape(E, C, M)                           │   │
│  │                                                                           │   │
│  │  (Each device processes only its assigned expert)                         │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          │  expert_output: [E, C, M] - processed tokens                          │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                      all_to_all (backward)                                │   │
│  │                                                                           │   │
│  │  lax.all_to_all(expert_output, axis_name='ep', split_axis=0, concat_axis=0)│   │
│  │                                                                           │   │
│  │  Before: Each device has outputs FOR tokens from all devices              │   │
│  │  After:  Each device has outputs FROM all experts for its local tokens    │   │
│  │                                                                           │   │
│  │  Also all_to_all for returned metadata                                    │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          │  returned: [E, C, M]                                                  │
│          │  returned_metadata: token_indices, weights                            │
│          │                                                                       │
│          ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                         combine_outputs                                   │   │
│  │                                                                           │   │
│  │  Scatter expert outputs back to original token positions                  │   │
│  │                                                                           │   │
│  │  output: [tokens_local, M] = zeros                                        │   │
│  │                                                                           │   │
│  │  for expert_idx in 0..E:                                                  │   │
│  │    for slot_idx in 0..C:                                                  │   │
│  │      token_idx = token_indices[expert_idx, slot_idx]                      │   │
│  │      weight = weights[expert_idx, slot_idx]                               │   │
│  │      if token_idx >= 0:  # valid, not padding                             │   │
│  │        output[token_idx] += weight * returned[expert_idx, slot_idx]       │   │
│  │                                                                           │   │
│  │  (Implemented as nested lax.scan for efficiency)                          │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                       │
│          ▼                                                                       │
│  OUTPUT (per device)                                                             │
│  ──────                                                                          │
│  output: [tokens_local, M]                                                       │
│  tokens_per_expert: [E]                                                          │
│  dropped_per_expert: [E]                                                         │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Sharding Summary (Parallel Folding Path)

| Component | Tensor | Type | Shape | Sharding (in_specs/out_specs) |
|-----------|--------|------|-------|-------------------------------|
| shard_map | x_shard | act | `[tokens_local, M]` | `P('ep', None)` |
| shard_map | top_k_ids_shard | act | `[tokens_local, K]` | `P('ep', None)` |
| shard_map | top_k_weights_shard | act | `[tokens_local, K]` | `P('ep', None)` |
| shard_map | output | act | `[tokens_local, M]` | `P('ep', None)` |
| shard_map | tokens_per_expert | act | `[E]` | `P('ep')` |
| shard_map | dropped_per_expert | act | `[E]` | `P('ep')` |

---

## Dimension Key

| Symbol | Meaning | Default Value |
|--------|---------|---------------|
| B | batch | 8 (global_batch_size) |
| S | sequence length | 256 (max_seq_len) |
| M | model/embed dim | 64 (d_model) |
| E | num_experts | 8 |
| H | mlp/hidden dim | 256 (d_ff) |
| K | num_experts_per_tok | 2 (top_k) |
| C | capacity per expert per batch (dense path) | `ceil((S*K)/E * capacity_factor)` |
| C' | capacity per expert total (parallel folding) | `ceil((B*S*K)/E * capacity_factor)` |

Note: `B × C = C'` — same total slots, different organization.

---

## Parallel Folding Example

Consider: 8 tokens across 2 devices, 4 experts, top-k=1, capacity=2

### Initial State

```
Device 0 has tokens: [T0, T1, T2, T3]
Device 1 has tokens: [T4, T5, T6, T7]

Routing assignments:
  T0 → Expert 0    T4 → Expert 1
  T1 → Expert 1    T5 → Expert 2
  T2 → Expert 2    T6 → Expert 3
  T3 → Expert 3    T7 → Expert 0
```

### Step 1: pack_tokens (per device)

```
Device 0 sendbuf [E=4, C=2, M]:
  Expert 0: [T0, _]
  Expert 1: [T1, _]
  Expert 2: [T2, _]
  Expert 3: [T3, _]

Device 1 sendbuf [E=4, C=2, M]:
  Expert 0: [T7, _]
  Expert 1: [T4, _]
  Expert 2: [T5, _]
  Expert 3: [T6, _]
```

### Step 2: all_to_all (forward)

Tokens are redistributed so each device receives tokens for its local experts.

With `split_axis=0, concat_axis=0`:
- Device 0 (owns experts 0,1) receives expert 0,1 slices from both devices
- Device 1 (owns experts 2,3) receives expert 2,3 slices from both devices

```
Device 0 recvbuf [E=4, C=2, M]:
  From Device 0: Expert 0: [T0, _], Expert 1: [T1, _]
  From Device 1: Expert 0: [T7, _], Expert 1: [T4, _]
  → Concatenated: [T0, _, T7, _, T1, _, T4, _] reshaped to [4, 2, M]

Device 1 recvbuf [E=4, C=2, M]:
  From Device 0: Expert 2: [T2, _], Expert 3: [T3, _]
  From Device 1: Expert 2: [T5, _], Expert 3: [T6, _]
  → Concatenated: [T2, _, T5, _, T3, _, T6, _] reshaped to [4, 2, M]
```

### Step 3: Expert Computation

```
Device 0: expert_fn(expert_id=0, [T0, T7, T1, T4, ...]) → processed
Device 1: expert_fn(expert_id=1, [T2, T5, T3, T6, ...]) → processed
```

### Step 4: all_to_all (backward)

Results are redistributed back to original token owners.

### Step 5: combine_outputs

Each device scatters results back to token positions using stored `token_indices` and applies `weights`.

---

## Comparison: Dense Einsum vs Parallel Folding

| Aspect | Dense Einsum | Parallel Folding |
|--------|--------------|------------------|
| **Token Movement** | Implicit via einsum contraction | Explicit via all_to_all |
| **Communication** | Compiler-inserted (GSPMD) | Explicit all_to_all in shard_map |
| **Intermediate Shape** | `[E, B, C, M]` | `[E, C', M]` (batch folded) |
| **Expert Computation** | Batched einsum over all experts | Per-device expert_fn call |
| **Mask Format** | `[B, S, E, C]` bool/float | `DispatchMetadata` with indices |
| **Memory** | Compiler-managed sharding of `[E, B, C, M]` | Explicit: each device packs/unpacks only its token shard |

**Note on capacity (C vs C')**: These represent different quantities:
- **C** (dense): capacity per expert *per batch item* = `ceil((S × K) / E × capacity_factor)`
- **C'** (parallel folding): capacity per expert *across all tokens* = `ceil((B × S × K) / E × capacity_factor)`

Total slots per expert are equal (`B × C = C'`), but organized differently. Dense keeps batch explicit; parallel folding folds batch into the capacity dimension.

### When to Use Which

**Dense Einsum** (MoEFFN):
- Simpler implementation, easier to reason about
- Relies on compiler (GSPMD) to optimize communication
- Matches MaxText's dense_matmul pattern
- Good when compiler produces efficient communication patterns

**Parallel Folding** (moe_dispatcher):
- Explicit control over communication (all_to_all placement and timing)
- Easier to debug/profile since communication is visible in code
- Better when compiler-generated patterns are suboptimal
- Avoids large mask contractions; tokens indexed directly into send buffer
- Tradeoff: must transmit metadata (token_indices, weights) alongside embeddings

### Relationship to ragged_all_to_all

The parallel folding dispatcher is essentially a manual implementation of what `jax.lax.ragged_all_to_all` provides as a primitive (used in MaxText's sparse_matmul path):

| Aspect | ragged_all_to_all | Parallel Folding |
|--------|-------------------|------------------|
| Message sizes | Variable (offsets + sizes) | Fixed buffers with padding |
| Metadata | Built into primitive | Separate all_to_all calls |
| Collectives | Single call | Multiple (data + metadata) |
| Efficiency | No wasted bandwidth | Transmits padding zeros |

Parallel folding can be viewed as a "portable" or "educational" version that works without ragged_all_to_all support, at the cost of some efficiency.

---

## Formal Specification (Machine-Readable)

### pack_tokens Algorithm

```
def pack_tokens(x, top_k_ids, top_k_weights, num_experts, capacity):
    # x: [T, M], top_k_ids: [T, K], top_k_weights: [T, K]

    sendbuf = zeros[E, C, M]
    token_indices = full[E, C](-1)  # -1 = padding
    weights = zeros[E, C]
    expert_counts = zeros[E]

    # Flatten assignments: process all (token, k) pairs sequentially
    for token_idx in 0..T:
        for k_idx in 0..K:
            expert_id = top_k_ids[token_idx, k_idx]
            weight = top_k_weights[token_idx, k_idx]
            pos = expert_counts[expert_id]

            if pos < capacity:
                sendbuf[expert_id, pos] = x[token_idx]
                token_indices[expert_id, pos] = token_idx
                weights[expert_id, pos] = weight
                expert_counts[expert_id] += 1
            # else: dropped

    return sendbuf, DispatchMetadata(token_indices, weights, ...)
```

### all_to_all Semantics

```
all_to_all(x, axis_name='ep', split_axis=0, concat_axis=0, tiled=True):
    # x: [E, C, M] on each device
    # E is split across devices (E/num_devices per device)

    # Each device i sends x[j, :, :] to device j
    # Each device j receives from device i and concatenates along axis 0

    # Result: [E, C, M] where:
    #   - Dimension 0 now indexes "source device" not "destination expert"
    #   - Device j has all data destined for expert j from all devices
```

### combine_outputs Algorithm

```
def combine_outputs(returned, metadata, num_tokens, d_model):
    # returned: [E, C, M] - expert outputs
    # metadata.token_indices: [E, C] - original token index, -1 for padding
    # metadata.weights: [E, C] - routing weights

    output = zeros[num_tokens, d_model]

    for expert_idx in 0..E:
        for slot_idx in 0..C:
            token_idx = metadata.token_indices[expert_idx, slot_idx]
            weight = metadata.weights[expert_idx, slot_idx]

            if token_idx >= 0:  # not padding
                output[token_idx] += weight * returned[expert_idx, slot_idx]

    return output
```

### Invariants

```
1. Token indices are unique within each expert's slots:
   for all e: token_indices[e, :] has no duplicates (except -1 padding)

2. Capacity limits dropping:
   for all e: count(token_indices[e, :] >= 0) <= capacity

3. Weights sum to <= 1 per token (may be < 1 if dropped):
   for all t: sum over (e, c) where token_indices[e, c] == t of weights[e, c] <= 1.0

4. all_to_all is its own inverse:
   all_to_all(all_to_all(x)) == x  (with same params)
```

---

## Load Balancing Loss

The implementation uses Switch Transformer style auxiliary loss:

```
aux_loss = num_experts * sum(f_i * P_i)

where:
  f_i = fraction of tokens routed to expert i
  P_i = average routing probability to expert i

Minimized when:
  - f is uniform (all experts get equal tokens)
  - P is uniform (router doesn't prefer specific experts)
```

This encourages balanced expert utilization during training.
