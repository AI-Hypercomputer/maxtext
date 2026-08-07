# Technical Design Specification: Double-Buffered Layer-Wise Parameter & Optimizer State Host Offloading for Scanned MaxText Models

**Author**: main agent (orchestrating via `planning_swarm` protocol)  
**Moniker**: `optimizer_offloading`  
**Status**: APPROVED DESIGN PROPOSAL  
**Branch**: `mohit/opt-off-trial`  

---

## 1. Problem Statement & Root Cause Analysis

### Current Flaw in MaxText
In MaxText (`src/maxtext/utils/maxtext_utils.py`), enabling `optimizer_memory_host_offload=True` or `parameter_memory_host_offload=True` sets the memory placement of `state_mesh_shardings.opt_state` and `state_mesh_shardings.params` to `pinned_host` (CPU host RAM).

However, during execution in `src/maxtext/trainers/pre_train/train.py` (lines 476–499), MaxText executes:
```python
if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(
                lambda x: x.with_memory_kind(kind="device"),
                state_mesh_shardings.opt_state,
            ),
        )
    )
if config.parameter_memory_host_offload:
    state = state.replace(
        params=jax.device_put(
            state.params,
            jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
        )
    )
```
This forces **all parameters and the entire optimizer state** back to TPU HBM simultaneously before running `state.apply_gradients`. Consequently:
1. Peak TPU HBM allocation is identical to non-offloaded training.
2. Device memory savings are zero during the optimizer step.
3. Host memory is used only as staging, incurring unnecessary transfer overhead.

---

## 2. Exploration of Design Alternatives (Swarm Proposals)

### Option A: Double-Buffered Scanned Layer Loop with Interleaved Optax Update (RECOMMENDED)
- **Concept**:
  Scanned transformer decoder layers stack parameters and optimizer states along `param_scan_axis=0` shape `(num_layers, ...)`. 
  Instead of updating all layers at once, the optimizer update is chunked layer-by-layer inside a `jax.lax.scan` loop (or interleaved with backward pass execution).
- **Mechanism**:
  - Unscanned parameters (Embeddings, LM Head) remain on device (or are updated separately).
  - Scanned parameters (`params_scanned`) and optimizer states (`opt_state_scanned`) are stored in `pinned_host` memory.
  - A double-buffered `jax.lax.scan` loop processes layer $i$:
    1. **Prefetch**: Issue asynchronous `jax.device_put(params[i+1], device_sharding)` to move layer $i+1$ from host to TPU HBM.
    2. **Compute & Update**: Execute backward pass / Optax update for layer $i$ on TPU.
    3. **Offload**: Issue `jax.device_put(new_opt_state[i], host_sharding)` to stream the updated optimizer state slice back to host RAM.
- **Pros**: Directly matches MaxText's scanned decoder architecture; lowest peak HBM overhead ($1$ layer instead of $N$ layers); simple double-buffering via JAX scan carry / scheduling groups.
- **Cons**: Requires separating scanned vs non-scanned parameters in `train.py`.

### Option B: Optax Piecewise Optimizer Wrapper (`piecewise_optimizer`)
- **Concept**:
  Wrap the Optax gradient transformation (`tx`) in a custom piecewise optimizer wrapper (inspired by `//depot/google3/experimental/users/davelacey/host_offload_training_example/piecewise_optimizer.py`).
- **Mechanism**:
  - The wrapper splits `state.params` and `state.opt_state` into sub-trees along `param_scan_axis=0`.
  - Applies `tx.update` sequentially over layer slices inside the optimizer transformation.
- **Pros**: Encapsulated within Optax interface.
- **Cons**: Optimizer state update still happens after full backward pass completes, which means all gradients for all layers must be held in HBM at the same time unless integrated into backward pass.

### Option C: Custom VJP Layer Pipeline (`custom_vjp`)
- **Concept**:
  Define a custom VJP for the full scanned decoder stack (similar to `src/maxtext/models/deepseek_batchsplit.py`).
- **Mechanism**:
  - In `process_all_layers_bwd`, compute layer $i$ gradient and immediately execute Optax update for layer $i$ before releasing layer $i$ activations.
- **Pros**: Maximum overlap between backward gradient computation and host memory streaming.
- **Cons**: Complex custom VJP setup; high risk of breaking model genericness across Flax Linen and NNX.

---

## 3. Deliberation & Selected Architectural Paradigm

**Selected Approach: Option A (Double-Buffered Scanned Layer Loop with Interleaved Optax Update)**

Option A is selected because:
1. MaxText's primary scalability for large models relies on scanned layers (`param_scan_axis=0`).
2. Parameters and optimizer states for scanned decoder layers are uniform tensors of shape `(num_layers, ...)`, making slice indexing trivial.
3. It integrates cleanly with both Flax Linen and NNX scanned layer mechanisms.

---

## 4. Detailed Technical Specification (Option A)

### 4.1 Parameter and Optimizer State Partitioning
State PyTrees are partitioned into two groups:
1. **Unscanned PyTree** ($P_{unscanned}, O_{unscanned}$): Embedding, final LayerNorm, LM Head. Kept on TPU HBM (`kind="device"`).
2. **Scanned PyTree** ($P_{scanned}, O_{scanned}$): Decoder block layers stacked along dimension 0. Placed in CPU Host RAM (`kind="pinned_host"`).

```python
def partition_scanned_state(state, param_scan_axis=0):
    """Separates scanned decoder parameters/opt_state from unscanned parameters."""
    # Leaves with leading dimension equal to num_layers and marked with param_scan_axis
    scanned_params, unscanned_params = split_pytree_by_scan_axis(state.params, param_scan_axis)
    scanned_opt, unscanned_opt = split_pytree_by_scan_axis(state.opt_state, param_scan_axis)
    return (scanned_params, unscanned_params), (scanned_opt, unscanned_opt)
```

### 4.2 Interleaved Double-Buffered Scan Loop

For updating scanned layers:
```python
def update_scanned_layers_double_buffered(scanned_params_host, scanned_opt_host, scanned_grads_device, tx, mesh_shardings):
    """
    Performs layer-by-layer Optax updates with double-buffered host<->device transfers.
    
    scanned_params_host: PyTree of shape (N, ...) on pinned_host
    scanned_opt_host: PyTree of shape (N, ...) on pinned_host
    scanned_grads_device: PyTree of shape (N, ...) on device
    """
    num_layers = jax.tree_util.tree_leaves(scanned_params_host)[0].shape[0]

    def scan_body(carry, i):
        # carry contains pre-fetched (params_i, opt_state_i) on device
        params_i_device, opt_state_i_device = carry

        # 1. Prefetch layer i+1 params & opt_state to device asynchronously
        next_i = jnp.minimum(i + 1, num_layers - 1)
        params_next_device = jax.tree.map(
            lambda x: jax.device_put(x[next_i], mesh_shardings.device_param_slice),
            scanned_params_host
        )
        opt_next_device = jax.tree.map(
            lambda x: jax.device_put(x[next_i], mesh_shardings.device_opt_slice),
            scanned_opt_host
        )

        # 2. Extract layer i gradients
        grads_i_device = jax.tree.map(lambda g: g[i], scanned_grads_device)

        # 3. Compute Optax update for layer i on device
        updates_i, new_opt_state_i = tx.update(grads_i_device, opt_state_i_device, params_i_device)
        new_params_i = optax.apply_updates(params_i_device, updates_i)

        # 4. Transfer updated opt_state_i and new_params_i back to pinned_host
        new_opt_state_i_host = jax.tree.map(
            lambda x: jax.device_put(x, mesh_shardings.host_opt_slice),
            new_opt_state_i
        )
        new_params_i_host = jax.tree.map(
            lambda x: jax.device_put(x, mesh_shardings.host_param_slice),
            new_params_i
        )

        next_carry = (params_next_device, opt_next_device)
        output_slice = (new_params_i_host, new_opt_state_i_host)
        return next_carry, output_slice

    # Prime pipeline with layer 0
    params_0_device = jax.tree.map(lambda x: jax.device_put(x[0], mesh_shardings.device_param_slice), scanned_params_host)
    opt_0_device = jax.tree.map(lambda x: jax.device_put(x[0], mesh_shardings.device_opt_slice), scanned_opt_host)
    init_carry = (params_0_device, opt_0_device)

    _, (updated_scanned_params, updated_scanned_opt) = jax.lax.scan(
        scan_body, init_carry, jnp.arange(num_layers)
    )
    return updated_scanned_params, updated_scanned_opt
```

### 4.3 Integration in `train.py`
In `src/maxtext/trainers/pre_train/train.py`:
- Replace full `jax.device_put` calls on `state.opt_state` and `state.params` with `update_scanned_layers_double_buffered`.
- Unscanned weights (Embeddings / Heads) are updated directly on device using standard `tx.update`.

---

## 5. Verification & Test Plan

1. **Memory Profiling Test**:
   - Compare peak HBM memory usage using `jax.live_arrays()` / XLA HLO memory stats between non-offloaded baseline and double-buffered scanned offloading.
   - Target: Peak HBM usage reduction proportional to $(N-1)/N \times \text{Optimizer State Size}$.

2. **Numerical Equivalence Unit Test**:
   - Run 10 training steps with `parameter_memory_host_offload=False` vs `parameter_memory_host_offload=True` & `optimizer_memory_host_offload=True`.
   - Assert `jnp.allclose(loss_offload, loss_baseline, atol=1e-6)`.

3. **Compiler Integration Test**:
   - Verify `train_compile_test.py` passes without XLA compilation errors or layout mismatch issues.

---

## 6. Approval & Execution Flow

As per project rules (`AGENTS.md`):
1. Design proposed by orchestrator (`planning_swarm`).
2. Design proposal presented to user for explicit approval before code implementation.
3. Upon user approval, coding tasks will be dispatched to subagent `tpu-raiden-coder`.
