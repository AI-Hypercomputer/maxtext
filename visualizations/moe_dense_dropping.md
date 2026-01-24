# MoE Dense Matmul - Token Dropping, Training (`capacity_factor > 0`)

Source: `src/MaxText/layers/moe.py`
- `RoutedMoE.__call__`: lines 1930-1968
- `GateLogit`: lines 161-298
- `dense_matmul`: lines 1659-1851

## Overview

When `capacity_factor > 0` and `model_call_mode != "inference"`, the MoE layer uses dispatch/combine masks to route tokens to experts. Tokens exceeding the expert capacity are dropped. This path uses explicit tensor reorganization via einsum to gather tokens per expert.

## Key Concepts

- **Capacity**: `expert_capacity = ceil(tokens_per_batch / num_experts) * capacity_factor`
- **Dispatch mask**: Gathers tokens to their assigned experts (with dropping)
- **Combine mask**: Scatters expert outputs back to token positions (with weights)

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
│  │  ╔═══════════════════════════════════════════════════════════════════╗    │   │
│  │  ║  output sharding (explicit mode only)                             ║    │   │
│  │  ║  ("activation_batch_no_exp", "activation_length_no_exp", None)    ║    │   │
│  │  ╚═══════════════════════════════════════════════════════════════════╝    │   │
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
│              dense_matmul (capacity_factor > 0, training)                        │
│                                                                                  │
│  INPUTS                                                                          │
│  ──────                                                                          │
│  inputs: [B, S, M]                                                               │
│  gate_logits: [B, S, E]               (E=num_experts)                            │
│                                                                                  │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  gate_logits sharding                                        ║                │
│  ║  ("activation_batch", "activation_norm_length", None)        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                |
│          │                                                                       │
│          ▼                                                                       │
│  ┌────────────────────────────────────┐                                          │
│  │           get_topk                 │                                          │
│  │  gate_logits: [B, S, E]            │                                          │
│  │           ↓                        │                                          │
│  │  top_k_weights: [B, S, K]          │  (K=num_experts_per_tok)                 │
│  │  top_k_indices: [B, S, K]          │                                          │
│  └────────────────────────────────────┘                                          │
│          │                                                                       │
│          ├─────────────────────────────────────────────────┐                     │
│          │                                                 │                     │
│          ▼                                                 ▼                     │
│  ┌──────────────────────────────┐                 ┌─────────────────────────┐    │
│  │ reshape_and_update_weights   │                 │    generate_masks       │    │
│  │                              │                 │                         │    │
│  │ IN:  top_k_weights [B,S,K]   │                 │ IN: top_k_indices [B,S,K│    │
│  │      top_k_indices [B,S,K]   │                 │     routing_weights     │    │
│  │              ↓               │                 │              ↓          │    │
│  │ OUT: routing_weights [B,S,E] │                 │ dispatch_mask: [B,S,E,C]│    │
│  │                              │                 │ combine_mask:  [B,S,E,C]│    │
│  │ (scatter K routing weights   │────────────────►│                         │    │
│  │  into dense E-dim tensor)    │                 │ (C=expert_capacity)     │    │
│  │                              │                 │                         │    │
│  │ NOTE: routing_weights is an  │                 │ combine_mask contains   │    │
│  │ activation, not a model      │                 │ routing weights for     │    │
│  │ weight matrix                │                 │ non-dropped assignments │    │
│  └──────────────────────────────┘                 └─────────────────────────┘    │
│                                                            │                     │
│  ╔════════════════════════════════════════════════════════════════════════╗      │
│  ║  mask sharding                                                         ║      │
│  ║  dispatch_mask, combine_mask: [B, S, E, C]                             ║      │
│  ║  ("activation_batch", "activation_norm_length", None, None)            ║      │
│  ╚════════════════════════════════════════════════════════════════════════╝      │
│                                                            │                     │
│          ┌─────────────────────────────────────────────────┘                     │
│          │                                                                       │
│          ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                            DISPATCH                                     │     │
│  │                                                                         │     │
│  │  einsum: BSM, BSEC -> EBCM                                              │     │
│  │                                                                         │     │
│  │  inputs: [B, S, M]              dispatch_mask: [B, S, E, C]             │     │
│  │                    ↘            ↙                                       │     │
│  │                      dispatch: [E, B, C, M]                             │     │
│  │                                                                         │     │
│  │  (tokens gathered per expert, up to capacity C)                         │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  dispatch sharding                                           ║                │
│  ║  ("activation_exp", "activation_batch_no_exp", None,         ║                │
│  ║   "activation_embed")                                        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                │
│          │                                                                       │
│          ├──────────────────────────────────┬────────────────────────────────┐   │
│          │                                  │                                │   │
│          ▼                                  ▼                                │   │
│  ┌─────────────────────────┐       ┌─────────────────────────┐               │   │
│  │        wi_0             │       │        wi_1             │               │   │
│  │                         │       │                         │               │   │
│  │ einsum: EBCM,EMH -> EBCH│       │ einsum: EBCM,EMH -> EBCH│               │   │
│  │                         │       │                         │               │   │
│  │ w0_kernel: [E, M, H]    │       │ w1_kernel: [E, M, H]    │               │   │
│  │ sharding:               │       │ sharding:               │               │   │
│  │  ("exp", None, "mlp")   │       │  ("exp", None, "mlp")   │               │   │
│  └─────────────────────────┘       └─────────────────────────┘               │   │
│          │                                  │                                │   │
│  ╔═══════════════════════════╗     ╔═══════════════════════════╗             │   │
│  ║ layer_w0 sharding         ║     ║ layer_w1 sharding         ║             │   │
│  ║ ("activation_exp",        ║     ║ ("activation_exp",        ║             │   │
│  ║  "activation_batch_no_exp"║     ║  "activation_batch_no_exp"║             │   │
│  ║  None, "activation_mlp")  ║     ║  None, "activation_mlp")  ║             │   │
│  ╚═══════════════════════════╝     ╚═══════════════════════════╝             │   │
│          │                                  │                                │   │
│          ▼                                  ▼                                │   │
│  layer_w0: [E, B, C, H]            layer_w1: [E, B, C, H]                    │   │
│          │                                  │                                │   │
│          ▼                                  │                                │   │
│  ┌─────────────────────────┐               │                                 │   │
│  │  activation (e.g. silu) │               │                                 │   │
│  └─────────────────────────┘               │                                 │   │
│          │                                  │                                │   │
│          └──────────────┬───────────────────┘                                │   │
│                         ▼                                                    │   │
│                 ┌───────────────────┐                                        │   │
│                 │   layer_w0 * w1   │  (element-wise multiply)               │   │
│                 └───────────────────┘                                        │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 layer_multiply: [E, B, C, H]                                 │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 ┌─────────────────────────┐                                  │   │
│                 │          wo             │                                  │   │
│                 │                         │                                  │   │
│                 │ einsum: EBCH,EHM -> EBCM│                                  │   │
│                 │                         │                                  │   │
│                 │ wo_kernel: [E, H, M]    │                                  │   │
│                 │ sharding:               │                                  │   │
│                 │  ("exp", "mlp", None)   │                                  │   │
│                 └─────────────────────────┘                                  │   │
│                         │                                                    │   │
│                 ╔═══════════════════════════════════════════════╗            │   │
│                 ║ intermediate_layer sharding                   ║            │   │
│                 ║ ("activation_exp", "activation_batch_no_exp", ║            │   │
│                 ║  None, "activation_embed")                    ║            │   │
│                 ╚═══════════════════════════════════════════════╝            │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 intermediate_layer: [E, B, C, M]                             │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │   │
│  │                            COMBINE                                      │ │   │
│  │                                                                         │ │   │
│  │  einsum: EBCM, BSEC -> BSM                                              │ │   │
│  │                                                                         │ │   │
│  │  intermediate_layer: [E,B,C,M]   combine_mask: [B, S, E, C]             │ │   │
│  │                      ↘            ↙                                     │ │   │
│  │                        output: [B, S, M]                                │ │   │
│  │                                                                         │ │   │
│  │  (expert outputs scattered back to token positions, weighted)           │ │   │
│  └─────────────────────────────────────────────────────────────────────────┘ │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│  OUTPUT                                                                      │   │
│  ──────                                                                      │   │
│  output: [B, S, M]                                                           │   │
│                                                                              │   │
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
| C | expert_capacity | `ceil(S*K/E) * capacity_factor` |

## Sharding Summary

| Component | Tensor | Type | Shape | Sharding |
|-----------|--------|------|-------|----------|
| GateLogit | kernel | weight | `[M, E]` | `("embed", None)` |
| GateLogit | bias | weight | `[E]` | `(None,)` |
| GateLogit | output | act | `[B, S, E]` | `("activation_batch_no_exp", "activation_length_no_exp", None)` * |
| dense_matmul | gate_logits | act | `[B, S, E]` | `("activation_batch", "activation_norm_length", None)` |
| dense_matmul | dispatch/combine_mask | act | `[B, S, E, C]` | `("activation_batch", "activation_norm_length", None, None)` |
| dense_matmul | dispatch | act | `[E, B, C, M]` | `("activation_exp", "activation_batch_no_exp", None, "activation_embed")` |
| dense_matmul | w0/w1_kernel | weight | `[E, M, H]` | `("exp", None, "mlp")` |
| dense_matmul | layer_w0/w1 | act | `[E, B, C, H]` | `("activation_exp", "activation_batch_no_exp", None, "activation_mlp")` |
| dense_matmul | wo_kernel | weight | `[E, H, M]` | `("exp", "mlp", None)` |
| dense_matmul | intermediate_layer | act | `[E, B, C, M]` | `("activation_exp", "activation_batch_no_exp", None, "activation_embed")` |

\* GateLogit output sharding only in explicit shard mode

## Notes

- GateLogit computes router scores: which experts each token should be sent to
- `config.routed_score_func` applies activation (e.g. sigmoid) before bias
- `config.routed_bias` enables learnable load-balancing bias (DeepSeek V3 style)
- Kernel weights may be all-gathered via `maybe_all_gather_kernel_weight_in_expert_parallelism` before matmul
- Token dropping happens in `generate_masks`: tokens exceeding `expert_capacity` get zero mask values
- The dispatch einsum reorders dimensions from `[B, S, ...]` to `[E, B, C, ...]` for expert-parallel computation
- The combine einsum reverses this, also applying the routing weights

## Routing Example

Consider a simple example with:
- **4 tokens** (S=4)
- **4 experts** (E=4)
- **top-k = 2** (each token routed to 2 experts)
- **capacity_factor = 1.0**

### Step 1: Compute Expert Capacity

```
tokens_per_batch = S × K = 4 × 2 = 8
expert_capacity = ceil(8 / 4) × 1.0 = 2
```

Each expert can handle **at most 2 tokens**.

### Step 2: Router Assigns Experts

Suppose `get_topk` returns these assignments:

| Token | Assigned Experts | Weights (after softmax) |
|-------|------------------|-------------------------|
| T0    | Expert 1, Expert 2 | 0.6, 0.4 |
| T1    | Expert 1, Expert 3 | 0.7, 0.3 |
| T2    | Expert 1, Expert 0 | 0.5, 0.5 |
| T3    | Expert 2, Expert 3 | 0.8, 0.2 |

### Step 3: Count Tokens Per Expert (cumsum)

Process tokens in order, counting arrivals at each expert:

| Token | Expert 0 | Expert 1 | Expert 2 | Expert 3 |
|-------|----------|----------|----------|----------|
| T0    | 0        | **1**    | **1**    | 0        |
| T1    | 0        | **2**    | 1        | **1**    |
| T2    | **1**    | **3** ❌ | 1        | 1        |
| T3    | 1        | 3        | **2**    | **2**    |

Expert 1 receives its 3rd token at T2, but capacity is 2 → **T2's assignment to Expert 1 is dropped**.

### Step 4: Generate Masks

**dispatch_mask** has shape `[B=1, S=4, E=4, C=2]`.

Viewed as a tree E→C→S (with B=1 omitted), each leaf is the mask value for that (Expert, Slot, Token) combination:

```
dispatch_mask[E, C, S]:

Expert 0:
├── slot 0: [T0=0, T1=0, T2=1, T3=0]  → contains T2
└── slot 1: [T0=0, T1=0, T2=0, T3=0]  → empty

Expert 1:
├── slot 0: [T0=1, T1=0, T2=0, T3=0]  → contains T0
└── slot 1: [T0=0, T1=1, T2=0, T3=0]  → contains T1
    (T2 would be slot 2, but capacity=2, so DROPPED)

Expert 2:
├── slot 0: [T0=1, T1=0, T2=0, T3=0]  → contains T0
└── slot 1: [T0=0, T1=0, T2=0, T3=1]  → contains T3

Expert 3:
├── slot 0: [T0=0, T1=1, T2=0, T3=0]  → contains T1
└── slot 1: [T0=0, T1=0, T2=0, T3=1]  → contains T3
```

**combine_mask** has the same structure but with softmax weights instead of binary:

```
combine_mask[E, C, S]:

Expert 0:
├── slot 0: [T0=0, T1=0, T2=0.5, T3=0]  → T2 with weight 0.5
└── slot 1: [T0=0, T1=0, T2=0,   T3=0]

Expert 1:
├── slot 0: [T0=0.6, T1=0, T2=0, T3=0]  → T0 with weight 0.6
└── slot 1: [T0=0, T1=0.7, T2=0, T3=0]  → T1 with weight 0.7

Expert 2:
├── slot 0: [T0=0.4, T1=0, T2=0, T3=0]  → T0 with weight 0.4
└── slot 1: [T0=0, T1=0, T2=0, T3=0.8]  → T3 with weight 0.8

Expert 3:
├── slot 0: [T0=0, T1=0.3, T2=0, T3=0]  → T1 with weight 0.3
└── slot 1: [T0=0, T1=0, T2=0, T3=0.2]  → T3 with weight 0.2
```

### Step 5: Dispatch (Gather)

The dispatch einsum `BSM, BSEC -> EBCM` contracts over S (tokens), gathering embeddings into expert slots.

For each (expert E, slot C), it sums `inputs[S] * dispatch_mask[S, E, C]` over all tokens S.
Since dispatch_mask is binary with exactly one token per slot, this effectively copies:

```
dispatch[expert=1, batch=0, slot=0, :] = T0_embedding  (mask[T0, E1, C0] = 1)
dispatch[expert=1, batch=0, slot=1, :] = T1_embedding  (mask[T1, E1, C1] = 1)
dispatch[expert=0, batch=0, slot=0, :] = T2_embedding  (mask[T2, E0, C0] = 1)
dispatch[expert=2, batch=0, slot=0, :] = T0_embedding  (mask[T0, E2, C0] = 1)
dispatch[expert=2, batch=0, slot=1, :] = T3_embedding  (mask[T3, E2, C1] = 1)
...
```

Each expert now has a `[B, C, M]` tensor containing its assigned tokens.

### Step 6: Expert Computation

Each expert independently processes its assigned tokens through wi_0, wi_1, wo.

### Step 7: Combine (Scatter)

The combine einsum `EBCM, BSEC -> BSM` scatters results back, weighted:

```
output[T0] = 0.6 × Expert1_output[slot0] + 0.4 × Expert2_output[slot0]
output[T1] = 0.7 × Expert1_output[slot1] + 0.3 × Expert3_output[slot0]
output[T2] = 0.5 × Expert0_output[slot0] + 0.0 × (dropped!)  ← only 1 expert!
output[T3] = 0.8 × Expert2_output[slot1] + 0.2 × Expert3_output[slot1]
```

**T2 only gets output from Expert 0** because its Expert 1 assignment was dropped due to capacity.

### Key Insight

The capacity mechanism ensures bounded compute per expert, but tokens that exceed capacity get degraded routing (fewer experts contribute to their output). Higher `capacity_factor` reduces dropping but increases memory/compute.

---

## Formal Specification (Machine-Readable)

This section provides a precise specification of the dispatch/combine mask mechanism for automated processing.

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
  C = expert_capacity = ceil((S × K) / E) × capacity_factor
```

### Data Structures

```
top_k_indices: int[B, S, K]
  - For each token, the K expert indices it is routed to
  - Values in range [0, E)

top_k_weights: float[B, S, K]
  - Softmax routing weights for each token's K expert assignments
  - NOT a model weight matrix - this is an activation derived from gate_logits
  - Sum over K equals 1.0 for each token

dispatch_mask: bool[B, S, E, C]
  - dispatch_mask[b, s, e, c] = True iff token s is assigned to expert e at capacity slot c
  - At most one token per (e, c) slot
  - Tokens exceeding capacity have all-False entries for that expert

combine_mask: float[B, S, E, C]
  - Contains routing weights (not model weights) for combining expert outputs
  - combine_mask[b, s, e, c] = top_k_weights[b, s, k] if dispatch_mask[b, s, e, c] else 0.0
  - Where k is the index such that top_k_indices[b, s, k] == e
```

### Mask Generation Algorithm

```
for each batch b:
  for each expert e:
    slot_counter[e] = 0

  for each token s in sequence order:
    for each k in 0..K:
      e = top_k_indices[b, s, k]
      if slot_counter[e] < C:
        c = slot_counter[e]
        dispatch_mask[b, s, e, c] = True
        combine_mask[b, s, e, c] = top_k_weights[b, s, k]
        slot_counter[e] += 1
      else:
        # Token dropped for this expert (capacity exceeded)
        pass
```

### Einsum Operations

```
DISPATCH (gather tokens into expert slots):
  einsum: "BSM, BSEC -> EBCM"

  Semantics:
    dispatch[e, b, c, m] = sum over s of: inputs[b, s, m] × dispatch_mask[b, s, e, c]

  Since dispatch_mask has exactly one s=1 per (e,c) pair:
    dispatch[e, b, c, :] = inputs[b, s_assigned, :] where s_assigned is the token in slot c of expert e

COMBINE (scatter expert outputs back to tokens):
  einsum: "EBCM, BSEC -> BSM"

  Semantics:
    output[b, s, m] = sum over e,c of: expert_output[e, b, c, m] × combine_mask[b, s, e, c]

  Equivalent to:
    output[b, s, :] = sum over assigned experts e: weight[e] × expert_output[e, b, slot[s,e], :]
```

### Invariants

```
1. Each (expert, slot) pair contains at most one token:
   for all e, c: sum over s of dispatch_mask[b, s, e, c] <= 1

2. Each token appears in at most K expert slots (may be fewer if dropped):
   for all s: sum over e, c of dispatch_mask[b, s, e, c] <= K

3. Combine routing weights sum to <= 1.0 per token (< 1.0 if some assignments dropped):
   for all s: sum over e, c of combine_mask[b, s, e, c] <= 1.0

4. Slot assignment is first-come-first-served by token index:
   if dispatch_mask[b, s1, e, c1] and dispatch_mask[b, s2, e, c2] and s1 < s2:
     then c1 < c2 (earlier tokens get earlier slots)
```

### Token Dropping Condition

```
A token s loses its assignment to expert e when:
  - e = top_k_indices[b, s, k] for some k
  - AND count of tokens s' < s where e in top_k_indices[b, s', :] >= C

Effect: combine_mask[b, s, e, :] is all zeros for that expert
Result: Token's output is computed from fewer than K experts
```
