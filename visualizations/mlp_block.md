# MlpBlock Tensor Flow Visualization

Source: `src/MaxText/layers/linears.py`

## Overview

`MlpBlock` is the standard transformer feed-forward block. It consists of:
- Optional pre-normalization
- Up-projection(s) via `DenseGeneral` (`wi` layers)
- Activation function(s)
- Optional gating (element-wise multiply for gated activations like SwiGLU)
- Down-projection via `DenseGeneral` (`wo` layer)

## Tensor Flow (Gated Activation, e.g. SwiGLU)

This shows the common case with `activations=("silu", "linear")` and `fused_mlp=False`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MlpBlock                                       │
│                                                                             │
│  INPUT                                                                      │
│  ──────                                                                     │
│  inputs: [batch, seq, embed]                                                │
│          sharding: (none specified at entry)                                │
│                                                                             │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────┐  (optional, if use_pre_norm=True)                    │
│  │  mlp_layer_norm   │                                                      │
│  └───────────────────┘                                                      │
│          │                                                                  │
│          ├────────────────────────┬──────────────────────┐                  │
│          │                        │                      │                  │
│          ▼                        ▼                      │                  │
│  ┌───────────────────┐    ┌───────────────────┐          │                  │
│  │      wi_0         │    │      wi_1         │          │                  │
│  │  (DenseGeneral)   │    │  (DenseGeneral)   │          │                  │
│  │                   │    │                   │          │                  │
│  │ kernel: [embed,   │    │ kernel: [embed,   │          │                  │
│  │          mlp]     │    │          mlp]     │          │                  │
│  │ sharding:         │    │ sharding:         │          │                  │
│  │  ("embed", "mlp") │    │  ("embed", "mlp") │          │                  │
│  └───────────────────┘    └───────────────────┘          │                  │
│          │                        │                      │                  │
│          ▼                        ▼                      │                  │
│  x_0: [batch, seq, mlp]   x_1: [batch, seq, mlp]         │                  │
│          │                        │                      │                  │
│          ▼                        │                      │                  │
│  ┌───────────────────┐            │                      │                  │
│  │   silu(x_0)       │            │  (linear = identity) │                  │
│  └───────────────────┘            │                      │                  │
│          │                        │                      │                  │
│          └────────────┬───────────┘                      │                  │
│                       ▼                                  │                  │
│               ┌───────────────┐                          │                  │
│               │   x_0 * x_1   │  (element-wise multiply) │                  │
│               └───────────────┘                          │                  │
│                       │                                  │                  │
│                       ▼                                  │                  │
│               x: [batch, seq, mlp]                       │                  │
│                       │                                  │                  │
│                       ▼                                  │                  │
│               ┌───────────────┐                          │                  │
│               │    dropout    │                          │                  │
│               └───────────────┘                          │                  │
│                       │                                  │                  │
│                       ▼                                  │                  │
│  ╔═══════════════════════════════════════════╗           │                  │
│  ║  maybe_shard_with_logical                 ║           │                  │
│  ║  x: [batch, seq, mlp]                     ║           │                  │
│  ║  sharding: ("activation_batch",           ║           │                  │
│  ║             "activation_length_no_exp",   ║           │                  │
│  ║             "activation_mlp")             ║           │                  │
│  ╚═══════════════════════════════════════════╝           │                  │
│                       │                                  │                  │
│                       ▼                                  │                  │
│               ┌───────────────────┐                      │                  │
│               │       wo          │                      │                  │
│               │  (DenseGeneral)   │                      │                  │
│               │                   │                      │                  │
│               │ kernel: [mlp,     │                      │                  │
│               │          embed]   │                      │                  │
│               │ sharding:         │                      │                  │
│               │  ("mlp", "embed") │                      │                  │
│               └───────────────────┘                      │                  │
│                       │                                  │                  │
│                       ▼                                  │                  │
│  OUTPUT                                                  │                  │
│  ──────                                                  │                  │
│  output: [batch, seq, embed]                             │                  │
│          sharding: out_sharding (passed in)              │                  │
│                                                          │                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tensor Flow (Fused MLP)

When `fused_mlp=True`, the two up-projections are combined into a single kernel:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MlpBlock (fused_mlp=True)                          │
│                                                                             │
│  INPUT                                                                      │
│  ──────                                                                     │
│  inputs: [batch, seq, embed]                                                │
│                                                                             │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────────────────────┐                                        │
│  │             wi                  │                                        │
│  │        (DenseGeneral)           │                                        │
│  │                                 │                                        │
│  │ kernel: [embed, num_act, mlp]   │                                        │
│  │ sharding:                       │                                        │
│  │  ("embed", "num_activations",   │                                        │
│  │   "mlp")                        │                                        │
│  └─────────────────────────────────┘                                        │
│          │                                                                  │
│          ▼                                                                  │
│  x: [batch, seq, num_act, mlp]                                              │
│          │                                                                  │
│          ├──────────────────┬──────────────────┐                            │
│          │                  │                  │                            │
│          ▼                  ▼                  │                            │
│  x[:,:,0,:] ───► silu    x[:,:,1,:] (linear)   │                            │
│          │                  │                  │                            │
│          └────────┬─────────┘                  │                            │
│                   ▼                            │                            │
│           ┌───────────────┐                    │                            │
│           │   x_0 * x_1   │                    │                            │
│           └───────────────┘                    │                            │
│                   │                            │                            │
│                   ▼                            │                            │
│           (dropout + sharding + wo)            │                            │
│           (same as non-fused path)             │                            │
│                                                │                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Sharding Summary

| Tensor | Type | Shape | Sharding |
|--------|------|-------|----------|
| `wi.kernel` (non-fused) | weight | `[embed, mlp]` | `("embed", "mlp")` |
| `wi.kernel` (fused) | weight | `[embed, num_act, mlp]` | `("embed", "num_activations", "mlp")` |
| `wo.kernel` | weight | `[mlp, embed]` | `("mlp", "embed")` |
| intermediate (after dropout) | act | `[batch, seq, mlp]` | `("activation_batch", "activation_length_no_exp", "activation_mlp")` * |

\* Sharding varies by mode: prefill uses `"prefill_activation_length"`, EP_AS_CONTEXT training uses `"activation_batch_no_exp"`.
