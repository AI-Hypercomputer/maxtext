# MoE Dense Matmul - No Token Dropping (`capacity_factor <= 0`)

Source: `src/MaxText/layers/moe.py`
- `RoutedMoE.__call__`: lines 1930-1968
- `GateLogit`: lines 161-298
- `dense_matmul`: lines 1852-1899

## Overview

When `capacity_factor <= 0`, the MoE layer uses a simpler weight-sum approach without token dropping. Each token is processed by all experts it's routed to, and the outputs are combined via weighted sum.

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
│                     dense_matmul (capacity_factor <= 0)                          │
│                                                                                  │
│  INPUTS                                                                          │
│  ──────                                                                          │
│  inputs: [B, S, M]                                                               │
│  gate_logits: [B, S, E]               (E=num_experts)                            │
│                                                                                  │
│          │                                                                       │
│          ▼                                                                       │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  gate_logits sharding                                        ║                │
│  ║  ("activation_batch", "activation_norm_length", None)        ║                │
│  ╚══════════════════════════════════════════════════════════════╝
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
│          ▼                                                                       │
│  ┌─────────────────────────────────────────┐                                     │
│  │    reshape_and_update_weights          │                                     │
│  │                                        │                                     │
│  │  IN:  top_k_weights: [B, S, K]         │  (softmax routing weights)          │
│  │       top_k_indices: [B, S, K]         │  (which experts selected)           │
│  │                 ↓                      │                                     │
│  │  OUT: routing_weights: [B, S, E]       │  (dense tensor: K non-zero values   │
│  │                                        │   per token, rest are zeros)        │
│  │                                        │                                     │
│  │  NOTE: This is an activation, not a    │                                     │
│  │  model weight matrix. Used later in    │                                     │
│  │  weight_sum einsum to combine experts. │                                     │
│  └─────────────────────────────────────────┘                                          │
│          │                                                                       │
│          │                                                                       │
│  ╔══════════════════════════════════════════════════════════════╗                │
│  ║  inputs sharding                                             ║                │
│  ║  ("activation_batch", "activation_norm_length",              ║                │
│  ║   "activation_embed")                                        ║                │
│  ╚══════════════════════════════════════════════════════════════╝                │
│          │                                                                       │
│          ├──────────────────────────────────┬────────────────────────────────┐   │
│          │                                  │                                │   │
│          ▼                                  ▼                                │   │
│  ┌─────────────────────────┐       ┌─────────────────────────┐               │   │
│  │        wi_0             │       │        wi_1             │               │   │
│  │                         │       │                         │               │   │
│  │ einsum: BSM,EMH -> BSEH │       │ einsum: BSM,EMH -> BSEH │               │   │
│  │                         │       │                         │               │   │
│  │ w0_kernel: [E, M, H]    │       │ w1_kernel: [E, M, H]    │               │   │
│  │ sharding: wi_kernel_axes│       │ sharding: wi_kernel_axes│               │   │
│  │ (see table below)       │       │ (see table below)       │               │   │
│  └─────────────────────────┘       └─────────────────────────┘               │   │
│          │                                  │                                │   │
│          ▼                                  ▼                                │   │
│  layer_w0: [B, S, E, H]            layer_w1: [B, S, E, H]                    │   │
│  (H=mlp_dim)                       (+ optional bias)                         │   │
│          │                                  │                                │   │
│          ▼                                  │                                │   │
│  ┌─────────────────────────┐                │                                │   │
│  │  activation (e.g. silu) │                │                                │   │
│  └─────────────────────────┘                │                                │   │
│          │                                  │                                │   │
│          └──────────────┬───────────────────┘                                │   │
│                         ▼                                                    │   │
│                 ┌───────────────────┐                                        │   │
│                 │   layer_w0 * w1   │  (element-wise multiply)               │   │
│                 └───────────────────┘                                        │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 layer_multiply: [B, S, E, H]                                 │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 ┌─────────────────────────┐                                  │   │
│                 │          wo             │                                  │   │
│                 │                         │                                  │   │
│                 │ einsum: BSEH,EHM -> BSEM│                                  │   │
│                 │                         │                                  │   │
│                 │ wo_kernel: [E, H, M]    │                                  │   │
│                 │ sharding: wo_kernel_axes│                                  │   │
│                 └─────────────────────────┘                                  │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 intermediate_layer: [B, S, E, M]                             │   │
│                 (+ optional bias)                                            │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│                 ┌───────────────────────────────────────┐                    │   │
│                 │        weight_sum                     │                    │   │
│                 │                                       │                    │   │
│                 │ einsum: BSEM, BSE -> BSM              │                    │   │
│                 │         ↑      ↑                      │                    │   │
│                 │         │      └─ routing_weights     │                    │   │
│                 │         └─ expert outputs             │                    │   │
│                 │                                       │                    │   │
│                 │ routing_weights: [B, S, E]            │                    │   │
│                 │ (from reshape_and_update_weights)     │                    │   │
│                 └───────────────────────────────────────┘                    │   │
│                         │                                                    │   │
│                         ▼                                                    │   │
│  OUTPUT                                                                      │   │
│  ──────                                                                      │   │
│  output: [B, S, M]                                                           │   │
│                                                                              │   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Kernel Sharding Axes

The kernel sharding depends on configuration:

| Config | wi_kernel_axes | wo_kernel_axes |
|--------|----------------|----------------|
| Default | `("exp", "embed_no_exp", "mlp")` | `("exp", "mlp", "embed_no_exp")` |
| `shard_exp_on_fsdp=True` | `("embed_no_exp", None, "mlp")` | `("embed_no_exp", "mlp", None)` |
| `use_2d_fsdp_sharding=True` | `("embed_no_exp", "mlp", None)` | `("embed_no_exp", "mlp", None)` |

## Dimension Key

| Symbol | Meaning | Typical Size |
|--------|---------|--------------|
| B | batch | varies |
| S | sequence length | varies |
| M | model/embed dim | e.g. 4096 |
| E | num_experts | e.g. 8, 64 |
| H | mlp/hidden dim | e.g. 14336 |
| K | num_experts_per_tok | e.g. 2 |

## Sharding Summary

| Component | Tensor | Type | Shape | Sharding |
|-----------|--------|------|-------|----------|
| GateLogit | kernel | weight | `[M, E]` | `("embed", None)` |
| GateLogit | bias | weight | `[E]` | `(None,)` |
| GateLogit | output | act | `[B, S, E]` | `("activation_batch_no_exp", "activation_length_no_exp", None)` * |
| dense_matmul | gate_logits | act | `[B, S, E]` | `("activation_batch", "activation_norm_length", None)` |
| dense_matmul | inputs | act | `[B, S, M]` | `("activation_batch", "activation_norm_length", "activation_embed")` |
| RoutedMoE | w0/w1_kernel | weight | `[E, M, H]` | `wi_kernel_axes` (config-dependent) |
| RoutedMoE | wo_kernel | weight | `[E, H, M]` | `wo_kernel_axes` (config-dependent) |
| dense_matmul | routing_weights | act | `[B, S, E]` | `("activation_batch_no_exp", "activation_length_no_exp", None)` |

\* GateLogit output sharding only in explicit shard mode

## Notes

- GateLogit computes router scores: which experts each token should be sent to
- `config.routed_score_func` applies activation (e.g. sigmoid) before bias
- `config.routed_bias` enables learnable load-balancing bias (DeepSeek V3 style)
- No intermediate activation sharding constraints in the dense_matmul path
- `float32_weight_sum` config casts tensors to f32 before final einsum
- `activations_in_float32` config casts after each wi matmul
- Load balance loss computed separately (not shown) when `load_balance_loss_weight > 0`

## Routing Example

Consider a simple example with:
- **4 tokens** (S=4)
- **4 experts** (E=4)
- **top-k = 2** (each token routed to 2 experts)

### Step 1: Router Assigns Experts

`get_topk` returns assignments and weights:

| Token | top_k_indices | top_k_weights (softmax) |
|-------|---------------|-------------------------|
| T0    | [1, 2]        | [0.6, 0.4]              |
| T1    | [1, 3]        | [0.7, 0.3]              |
| T2    | [0, 1]        | [0.5, 0.5]              |
| T3    | [2, 3]        | [0.8, 0.2]              |

### Step 2: Scatter Routing Weights to Dense Tensor

`reshape_and_update_weights` converts sparse `[B, S, K]` to dense `[B, S, E]`:

```
routing_weights[B=0, :, :]:

         Expert 0  Expert 1  Expert 2  Expert 3
Token 0: [  0.0  ,   0.6   ,   0.4   ,   0.0   ]
Token 1: [  0.0  ,   0.7   ,   0.0   ,   0.3   ]
Token 2: [  0.5  ,   0.5   ,   0.0   ,   0.0   ]
Token 3: [  0.0  ,   0.0   ,   0.8   ,   0.2   ]
```

### Step 3: Broadcast Computation Across Experts

The einsum `BSM, EMH -> BSEH` computes all token-expert combinations:

```
layer_w0[b, s, e, :] = inputs[b, s, :] @ w0_kernel[e, :, :]
```

Every token is multiplied by every expert's weights, producing `[B, S, E, H]`.

### Step 4: Weight Sum

The final einsum `BSEM, BSE -> BSM` combines expert outputs:

```
output[T0] = 0.6 × Expert1_output[T0] + 0.4 × Expert2_output[T0]
output[T1] = 0.7 × Expert1_output[T1] + 0.3 × Expert3_output[T1]
output[T2] = 0.5 × Expert0_output[T2] + 0.5 × Expert1_output[T2]
output[T3] = 0.8 × Expert2_output[T3] + 0.2 × Expert3_output[T3]
```

### Key Insight

Unlike the token-dropping path, **every token always gets its full top-k experts**. The tradeoffs are:

**Computation**: `O(S × E)` rather than `O(E × C)` where `C = capacity` is typically much smaller than `S`. We compute all experts for all tokens, then mask via the weight sum.

**Memory**: Intermediate activations are larger:
- `layer_w0, layer_w1`: `[B, S, E, H]` vs `[E, B, C, H]`
- `intermediate_layer`: `[B, S, E, M]` vs `[E, B, C, M]`

For example, with S=2048, E=64, C=64 (capacity_factor=1.0, K=2):
- Weight-sum: `B × 2048 × 64 × H` = `B × 131072 × H`
- Token-dropping: `64 × B × 64 × H` = `B × 4096 × H` (32× smaller)

This path is simpler but less efficient for large S or E.

---

## Formal Specification (Machine-Readable)

This section provides a precise specification of the weight-sum mechanism for automated processing.

### Definitions

```
INPUTS:
  B = batch size
  S = sequence length (number of tokens)
  E = number of experts
  K = num_experts_per_tok (top-k)
  M = model/embedding dimension
  H = MLP hidden dimension
```

### Data Structures

```
top_k_indices: int[B, S, K]
  - For each token, the K expert indices it is routed to
  - Values in range [0, E)

top_k_weights: float[B, S, K]
  - Softmax weights for each token's K expert assignments
  - Sum over K equals 1.0 for each token

routing_weights: float[B, S, E]
  - Dense routing weight tensor created by reshape_and_update_weights
  - NOT a model parameter - this is an activation derived from gate_logits
  - routing_weights[b, s, e] = top_k_weights[b, s, k] if top_k_indices[b, s, k] == e else 0.0
  - Exactly K non-zero values per token (row)
```

### Routing Weight Scattering Algorithm

```
routing_weights = zeros[B, S, E]

for each batch b:
  for each token s:
    for each k in 0..K:
      e = top_k_indices[b, s, k]
      routing_weights[b, s, e] = top_k_weights[b, s, k]
```

### Einsum Operations

```
UP-PROJECTION (wi_0, wi_1):
  einsum: "BSM, EMH -> BSEH"

  Semantics:
    layer_w[b, s, e, h] = sum over m of: inputs[b, s, m] × kernel[e, m, h]

  Note: Computes ALL token-expert pairs (broadcast), not just routed ones

DOWN-PROJECTION (wo):
  einsum: "BSEH, EHM -> BSEM"

  Semantics:
    intermediate[b, s, e, m] = sum over h of: layer_multiply[b, s, e, h] × wo_kernel[e, h, m]

WEIGHT SUM (final combination):
  einsum: "BSEM, BSE -> BSM"

  Semantics:
    output[b, s, m] = sum over e of: intermediate[b, s, e, m] × routing_weights[b, s, e]

  Since routing_weights has K non-zero entries per token:
    output[b, s, :] = sum over assigned experts e: routing_weights[b, s, e] × intermediate[b, s, e, :]
```

### Invariants

```
1. Each token has exactly K non-zero routing weights:
   for all b, s: count of (routing_weights[b, s, e] > 0) == K

2. Routing weights sum to 1.0 per token:
   for all b, s: sum over e of routing_weights[b, s, e] == 1.0

3. All expert computations are performed (broadcast):
   layer_w0, layer_w1, intermediate all have shape [B, S, E, ...]
   (no sparsity in intermediate tensors)

4. Final output selects via routing weight masking:
   Experts with routing_weights[b, s, e] == 0 contribute nothing to output[b, s, :]
```

### Comparison to Token-Dropping Path

```
Weight-Sum (this path):
  - Computation: O(B × S × E × H) for each MLP layer
  - Memory: O(B × S × E × H) for intermediate activations
  - All tokens get exactly K experts (no dropping)

Token-Dropping (capacity_factor > 0):
  - Computation: O(B × E × C × H) where C = capacity << S typically
  - Memory: O(B × E × C × H) for intermediate activations
  - Tokens may get fewer than K experts if capacity exceeded
```
