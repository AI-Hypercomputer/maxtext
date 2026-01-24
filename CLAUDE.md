# MaxText Project Context

## Workflow Rules

- **Do not commit, push, or amend commits** without explicit user request
- Wait for user to ask before any git operations that modify history or remote

## Current Task: Weight Tensor Flow Visualization

Goal: Create simple descriptions and visualizations showing how weight tensors flow through various model implementations.

### Models of Interest
- Standard transformer models (dense)
- MoE (Mixture of Experts) models with three implementations:
  1. `dense_matmul` - dense matrix multiplication approach
  2. `sparse_matmul` (Megablox) - using the Megablox kernel
  3. `sparse_matmul` (ragged_all_to_all) - using `jax.lax.ragged_all_to_all`

### Key Considerations
- Models are designed for multi-device training with complex sharding rules
- Focus on understanding tensor shapes and how they transform through the forward pass
- Document sharding annotations and their implications

### Deliverables
- Descriptions of tensor flow through each model variant
- Visualizations of weight tensor transformations

## Visualization Format

When creating tensor flow visualizations, include the following for each tensor:

1. **Name** - the variable/parameter name
2. **Type** - `weight` (model parameters) or `act` (activations)
3. **Shape** - tensor dimensions using semantic names (e.g., `[batch, seq, embed]`)
4. **Sharding** - logical sharding axes, if specified

### Sharding Rules
- **Weight tensors**: Always have sharding specified via `kernel_axes` or similar parameter
- **Activation tensors**: Only show sharding when explicitly constrained via:
  - `maybe_shard_with_logical`
  - `maybe_shard_with_name`
  - `jax.lax.with_sharding_constraint`
  - Similar sharding constraint wrappers

### Diagram Conventions
- Use ASCII box diagrams showing data flow
- Mark modules/layers with boxes (`┌───┐`)
- Mark explicit sharding constraints with double-line boxes (`╔═══╗`)
- Show tensor shapes and shardings inline with flow
- Indicate optional/conditional paths (e.g., "if use_pre_norm=True")

### Describing High-Dimensional Tensors

For tensors with 3+ dimensions (especially masks, indices, or sparse structures), use **tree notation** rather than trying to show the full tensor or describe it from one axis's perspective.

**Example**: A mask with shape `[B, S, E, C]` is hard to visualize. Instead:
1. Fix batch (B=1) to reduce dimensions
2. Choose a natural hierarchy for the remaining axes (e.g., E→C→S)
3. Show as a tree with values at leaves

```
mask[E, C, S]:

Expert 0:
├── slot 0: [T0=0, T1=0, T2=1, T3=0]  → contains T2
└── slot 1: [T0=0, T1=0, T2=0, T3=0]  → empty

Expert 1:
├── slot 0: [T0=1, T1=0, T2=0, T3=0]  → contains T0
└── slot 1: [T0=0, T1=1, T2=0, T3=0]  → contains T1
```

This makes it clear what the tensor represents: "for each (expert, slot), which tokens are assigned?"

**Guidelines**:
- Choose hierarchy based on how the tensor is consumed (e.g., if einsum contracts over S, show S as leaves)
- Annotate each leaf with semantic meaning (e.g., "→ contains T2")
- For weighted masks, show weights instead of binary values

### Machine-Readable Specifications

For complex mechanisms (e.g., capacity masking, attention patterns), include a **Formal Specification** section with:

1. **Definitions**: All dimension variables with precise meanings
2. **Data Structures**: Shape and semantics of each tensor
3. **Algorithm**: Pseudocode showing the computation
4. **Einsum Operations**: Both the einsum string and semantic interpretation
5. **Invariants**: Formal properties that always hold

This dual format (visual examples for humans + formal specs for machines) ensures the documentation is useful for both human readers and LLM agents.

### Output Location
Visualizations are stored in `/visualizations/*.md`

## MoE Layer Architecture

Key classes in `src/MaxText/layers/moe.py`:

### Class Hierarchy
- `RoutedMoE` - Main MoE block, calls GateLogit then dense_matmul or sparse_matmul
- `GateLogit` - Computes router scores (which experts each token goes to)
- `RoutedAndSharedMoE` - Combines routed experts with shared experts (DeepSeek style)

### RoutedMoE Flow
1. `inputs: [B, S, M]` → `GateLogit` → `gate_logits: [B, S, E]`
2. Extract kernels `wi_0, wi_1, wo` from params
3. Branch on `config.sparse_matmul`:
   - `False` → `dense_matmul`
   - `True` → `sparse_matmul`

### dense_matmul Branches
- **`capacity_factor > 0`**: Token dropping with dispatch/combine masks
- **`capacity_factor <= 0`**: Simple weight-sum approach (no dropping)

### Key Config Parameters
| Parameter | Effect |
|-----------|--------|
| `capacity_factor` | >0 enables token dropping, <=0 uses weight-sum |
| `sparse_matmul` | True uses Megablox or ragged_all_to_all kernels |
| `routed_bias` | Enables learnable load-balancing bias (DeepSeek V3) |
| `routed_score_func` | Activation before bias (e.g. "sigmoid") |
| `shard_exp_on_fsdp` | Changes kernel sharding axes |
| `use_2d_fsdp_sharding` | Alternative 2D FSDP sharding |

### Kernel Sharding (default config)
| Kernel | Shape | Sharding |
|--------|-------|----------|
| GateLogit.kernel | `[M, E]` | `("embed", None)` |
| wi_0, wi_1 | `[E, M, H]` | `("exp", "embed_no_exp", "mlp")` |
| wo | `[E, H, M]` | `("exp", "mlp", "embed_no_exp")` |

### Completed Visualizations
- `visualizations/mlp_block.md` - Standard MLP feed-forward block
- `visualizations/moe_dense_dropless.md` - MoE without token dropping
- `visualizations/moe_dense_dropping.md` - MoE with token dropping (training)
