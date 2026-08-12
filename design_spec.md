# MaxText Architecture Specification: Layer-Wise Parameter Host Offloading (NNX Backend)

**Target Branch:** `mohit/bharat_opt_off`  
**Author:** Principal JAX & MaxText Distributed Systems Architect  
**Date:** 2026-08-09  
**Status:** Approved Architectural Design Specification  

---

## 1. Executive Summary & Problem Statement

### 1.1 Context and Motivation
In ultra-large autoregressive transformer models (e.g., Qwen3-Next-80B, DeepSeek-V3/V4, Gemma4), storing the entire model parameter set in TPU High-Bandwidth Memory (HBM) alongside activations, KV caches, and optimizer states leads to severe memory exhaustion (OOM), especially under tensor parallel (TP), expert parallel (EP), and context parallel (CP) constraints where per-chip memory budgets are strictly bounded.

In MaxText's Flax NNX backend, repeated decoder layers are organized into stacked/scanned blocks. Crucially, during the forward and backward execution passes, only **one layer's parameter slice** is active at any given execution step. Storing the full stacked parameter tensor in TPU HBM is wasteful and restricts trainable parameter counts.

### 1.2 Core Architectural Principles & Invariants
This specification defines the exact **Layer-Wise Parameter Host Offloading** architecture for Flax NNX in MaxText:
1. **Scanned Block Parameters in Host Pinned Memory (`pinned_host`)**:
   - The master stacked parameter tensors (`decoder.scanned_blocks.local_layers.*`) reside in pinned host DRAM (`pinned_host` memory kind) across training steps.
   - During forward/backward execution, each layer slice is prefetched asynchronously into TPU device HBM (`device` memory kind) inside `scan_body`, executed, and immediately reclaimed.
   - During the optimizer step (`train_step`), parameters are staged into `device` HBM via compiler sharding constraints before `apply_gradients`, updated natively by Optax on TPU hardware, and constrained back to `pinned_host` immediately post-update.
2. **Optimizer States Exclusively in Device Memory (`device`)**:
   - First moments (`mu`), second moments (`nu`), step counter, and optimizer metadata remain permanently in TPU HBM (`memory_kind="device"`), sharded under Zero-1 / FSDP data parallelism.
   - **No optimizer host offloading** is performed, eliminating multi-gigabyte PCIe bandwidth bottlenecks during optimizer updates.
3. **Non-Scanned Parameters Exclusively in Device Memory (`device`)**:
   - Token embeddings (`decoder.token_embedder`), final normalizations (`decoder.decoder_norm`), unscanned remainder layers (`decoder.layers_remainder`), and global attention layers (`decoder.scanned_blocks.global_layer`) remain permanently in `device` HBM.
4. **Gradients Generated and Consumed in Device Memory (`device`)**:
   - Gradients (`grads`) produced by `jax.value_and_grad` are immediately normalized/constrained to `device` memory kind before any norm calculations or gradient clipping, and consumed directly by `state.apply_gradients(grads)` on TPU.
5. **Elimination of Nested JIT and Dynamic Compiles**:
   - Memory kind transitions are orchestrated entirely via first-class `jax.lax.with_sharding_constraint` compiler directives inside the top-level `jax.jit(train_step)`. All nested `jax.jit` invocations are strictly prohibited.

---

## 2. TrainStateNNX Component Hierarchy & Memory Kind Lifecycle Matrix

### 2.1 PyTree Structure of TrainStateNNX
In Flax NNX, [TrainStateNNX](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/common/train_state_nnx.py#L24-L57) unifies model weights and optimizer states into a single structured PyTree:
```
TrainStateNNX
├── model (nnx.State)
│   └── decoder
│       ├── token_embedder (Embeddings / LM Head)
│       │   └── embedding: nnx.Param
│       ├── scanned_blocks (Stacked Scannable Blocks)
│       │   ├── local_layers (GatedDeltaNet / Linear Attention) [SCANNED]
│       │   │   ├── attention: {in_proj_ba, in_proj_qkvz, out_proj, conv1d, A_log, dt_bias, norm}
│       │   │   ├── input_layernorm: {scale}
│       │   │   ├── mhc_attention: {mhc_norm, pre_*, post_*, res_*}
│       │   │   ├── mhc_mlp: {mhc_norm, pre_*, post_*, res_*}
│       │   │   ├── mlp: {routed_experts, shared_expert, shared_expert_gate}
│       │   │   └── post_attention_layernorm: {scale}
│       │   └── global_layer (Full Attention / Unscanned Global Block) [UNSCANNED]
│       │       ├── attention: {query, key, value, out, query_norm, key_norm}
│       │       └── ...
│       ├── layers_remainder (Remainder Block when num_layers % cycle != 0) [UNSCANNED]
│       ├── decoder_norm (Final RMSNorm) [UNSCANNED]
│       │   └── scale: nnx.Param
│       ├── logits_dense (Output Head) [UNSCANNED]
│       └── dropout / rngs: {aqt, dropout, params} (nnx.RngState / nnx.RngCount)
└── optimizer (nnx.State)
    ├── step: uint32
    └── opt_state (Optax State PyTree)
        ├── 0 (AdamW / Muon Momentum & Variance)
        │   ├── mu: PyTree matching model params (Zero-1 sharded)
        │   └── nu: PyTree matching model params (Zero-1 sharded)
        ├── 1 ...
        └── count: int32 / uint32
```

### 2.2 Complete Memory Kind Lifecycle Matrix

| Component | NNX PyTree Path Pattern | Physical Rank / Shape | Step Input Sharding (`in_shardings`) | Forward Pass (`scan_body`) | Backward Pass (Adjoint/Remat) | Gradient Generation (`grads`) | Pre-`apply_gradients` Staging | `apply_gradients` Execution | Post-`apply_gradients` Eviction | Step Output Sharding (`out_shardings`) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Scanned Block Parameters** | `model/decoder/scanned_blocks/local_layers/**` | Stacked `[N_b, N_l, ...]` | `pinned_host` | Prefetched per-slice to `device` HBM; evicted at iter end | Prefetched per-slice to `device` HBM; evicted at iter end | `device` (staged immediately post-VJP) | Staged to `device` via `with_sharding_constraint` | `device` (TPU ALU/MXU compute) | Evicted to `pinned_host` via `with_sharding_constraint` | `pinned_host` |
| **Global Layer Parameters** | `model/decoder/scanned_blocks/global_layer/**` | Unscanned `[...]` | `device` | Kept in `device` HBM | Kept in `device` HBM | `device` | `device` (no-op) | `device` | `device` (no-op) | `device` |
| **Remainder Layer Parameters** | `model/decoder/layers_remainder/**` | Unscanned `[...]` | `device` | Kept in `device` HBM | Kept in `device` HBM | `device` | `device` (no-op) | `device` | `device` (no-op) | `device` |
| **Embeddings & Logits Head** | `model/decoder/token_embedder/**`, `model/decoder/logits_dense/**` | `[V, D]`, `[D, V]` | `device` | Kept in `device` HBM | Kept in `device` HBM | `device` | `device` (no-op) | `device` | `device` (no-op) | `device` |
| **Normalizations** | `model/decoder/decoder_norm/**` | `[D]` | `device` | Kept in `device` HBM | Kept in `device` HBM | `device` | `device` (no-op) | `device` | `device` (no-op) | `device` |
| **RNG & Dropout State** | `model/decoder/dropout/rngs/**` | Scalar / `[2]` | `device` | Kept in `device` HBM | Kept in `device` HBM | N/A | `device` (no-op) | `device` | `device` (no-op) | `device` |
| **Optimizer 1st Moment (`mu`)** | `optimizer/opt_state/*/mu/**` | Matches model params (Zero-1 sharded) | `device` | Idle in `device` HBM | Idle in `device` HBM | N/A | `device` | `device` (Optax update) | `device` | `device` |
| **Optimizer 2nd Moment (`nu`)** | `optimizer/opt_state/*/nu/**` | Matches model params (Zero-1 sharded) | `device` | Idle in `device` HBM | Idle in `device` HBM | N/A | `device` | `device` (Optax update) | `device` | `device` |
| **Optimizer Step & Counters** | `optimizer/step`, `optimizer/opt_state/*/count` | Scalar | `device` | Idle in `device` HBM | Idle in `device` HBM | N/A | `device` | `device` (Scalar increment) | `device` | `device` |
| **Batch Input & Targets** | `batch/{inputs, targets, ...}` | `[B, S]` | `device` | Kept in `device` HBM | Kept in `device` HBM | N/A | N/A | N/A | N/A | N/A |

---

## 3. Detailed Architectural Workflow & Lifecycle Diagram

```mermaid
sequenceDiagram
    autonumber
    participant HostDRAM as Host Pinned DRAM (pinned_host)
    participant TPU_HBM as TPU HBM (device)
    participant TPU_Compute as TPU Matrix/Vector Engines (MXU/VPU)
    participant Optax as Optax Engine (apply_gradients)

    Note over HostDRAM,TPU_HBM: STEP INITIALIZATION
    HostDRAM->>TPU_HBM: Batch & Non-Scanned Params in HBM; Scanned Params in pinned_host
    
    rect rgb(230, 245, 255)
    Note over TPU_HBM,TPU_Compute: FORWARD PASS (Scan Loop)
    loop For each Layer / Block Slice
        HostDRAM->>TPU_HBM: Async DMA Prefetch Layer Slice i (pinned_host -> device)
        TPU_HBM->>TPU_Compute: Execute Layer Forward GEMM & Attention
        TPU_Compute->>TPU_HBM: Write Output Hidden States & Checkpoint Activations
        Note over TPU_HBM: Deallocate Layer Slice i weights from HBM (Ephemeral)
    end
    end

    rect rgb(255, 240, 240)
    Note over TPU_HBM,TPU_Compute: BACKWARD PASS (Adjoint VJP)
    loop For each Layer / Block Slice (Reverse)
        HostDRAM->>TPU_HBM: Async DMA Prefetch Layer Slice i for VJP / Remat
        TPU_Compute->>TPU_HBM: Accumulate Gradients (grads)
        Note over TPU_HBM: Deallocate Layer Slice i weights from HBM
    end
    Note over TPU_HBM: Ensure ALL raw_grads leaves constrained to device HBM
    end

    rect rgb(240, 255, 240)
    Note over HostDRAM,Optax: OPTIMIZER STAGING & UPDATE
    HostDRAM->>TPU_HBM: Stage Full Scanned Params: with_sharding_constraint(s.with_memory_kind("device"))
    Note over TPU_HBM,Optax: grads (device), model_params (device), opt_state (device)
    TPU_HBM->>Optax: state.apply_gradients(grads) on TPU hardware
    Optax->>TPU_HBM: Compute new mu, nu, updated_params in device HBM
    TPU_HBM->>HostDRAM: Evict Scanned Params: with_sharding_constraint(s.with_memory_kind("pinned_host"))
    Note over TPU_HBM: opt_state (mu, nu, step) and non-scanned params remain in TPU HBM
    end
```

---

## 4. Exact Sharding Transformation Contracts

### 4.1 Path Matching Contract: `is_scanned_block_param_path`
The system must unambiguously identify which tensors in the `TrainStateNNX` PyTree are stacked scanned block parameters versus non-scanned parameters or optimizer states.

```python
def is_scanned_block_param_path(path: tuple[Any, ...]) -> bool:
  """Returns True if the PyTree path corresponds strictly to a scanned block model parameter.
  
  Invariants:
  1. Must be under `model` (or root parameter container).
  2. Must contain scanned container identifiers ('scanned_blocks', 'local_layers', 'scanned_layers').
  3. Must NOT contain unscanned exclusions ('global_layer', 'layers_remainder', 'token_embedder',
     'decoder_norm', 'logits_dense').
  4. Must NOT contain optimizer state identifiers ('opt_state', 'optimizer', 'mu', 'nu', 'step', 'count', 'rngs').
  """
  path_keys = [str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in path]
  
  # Check if under optimizer hierarchy
  if any(k in path_keys for k in ("optimizer", "opt_state", "mu", "nu", "step", "count")):
    return False
    
  # Check if under RNG / dropout hierarchy
  if any(k in path_keys for k in ("rngs", "dropout", "key")):
    return False

  # Check if model parameter
  is_model = ("model" in path_keys) or (len(path_keys) > 0 and path_keys[0] != "optimizer")
  
  # Check for scanned containers
  has_scanned = any(k in path_keys for k in ("scanned_blocks", "local_layers", "scanned_layers", "layers"))
  
  # Check for unscanned exceptions
  is_unscanned_exclusion = any(
      k in path_keys
      for k in (
          "global_layer",
          "layers_remainder",
          "token_embedder",
          "embedding",
          "decoder_norm",
          "norm",
          "logits_dense",
      )
  )
  
  return is_model and has_scanned and not is_unscanned_exclusion
```

---

### 4.2 Sharding Setup Contract: `build_zero1_input_state_mesh_shardings` & `maybe_update_params_sharding_with_opt_nnx`

In [src/maxtext/utils/sharding.py](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/utils/sharding.py#L671-L809):

1. **`maybe_update_params_sharding_with_opt_nnx`**:
   - Zero-1 shards the optimizer states (`mu`, `nu`) over the data/fsdp mesh axes.
   - When updating `model_shardings` from `mu_lookup`, the existing `memory_kind` MUST be preserved:
     ```python
     def _update_model_var(path, var):
       if path in mu_lookup:
         new_s = mu_lookup[path]
         curr_s = getattr(var, "value", var)
         # If model parameter was tagged pinned_host, preserve pinned_host
         if hasattr(curr_s, "memory_kind") and curr_s.memory_kind == "pinned_host":
           new_s = new_s.with_memory_kind("pinned_host")
         elif is_scanned_block_param_path(path) and (config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False)):
           new_s = new_s.with_memory_kind("pinned_host")
         else:
           new_s = new_s.with_memory_kind("device")
         if isinstance(var, nnx.Variable):
           return var.replace(new_s)
         return new_s
       return var
     ```

2. **`build_zero1_input_state_mesh_shardings`**:
   - Maps over `state_mesh_shardings` using normalized path matching.
   - Strictly guarantees that:
     - All `state.model` parameters satisfying `is_scanned_block_param_path(path)` receive `memory_kind="pinned_host"`.
     - All other model parameters (embeddings, norms, global layers, remainder layers) receive `memory_kind="device"`.
     - All `state.optimizer` leaves (optimizer moments `mu`, `nu`, `step`, `count`) receive `memory_kind="device"`.

```python
def build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings):
  """Constructs the canonical input state shardings for TrainStateNNX under Zero-1 and Host Offload."""
  if not config.shard_optimizer_over_data and not (config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False)):
    return state_mesh_shardings

  def _to_str_path(path):
    return tuple(str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k) for k in path)

  param_leaves = list(
      jax.tree_util.tree_leaves_with_path(params_shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.sharding.Sharding)))
  )
  param_lookup = {_to_str_path(path): (v.value if hasattr(v, "value") else v) for path, v in param_leaves}

  def _update_sharding(path, curr_s):
    str_path = _to_str_path(path)
    lookup_path = str_path[1:] if len(str_path) > 0 and str_path[0] == "model" else str_path
    target_sharding = param_lookup.get(lookup_path, param_lookup.get(str_path, curr_s))
    
    # Apply memory_kind rules
    if hasattr(target_sharding, "with_memory_kind"):
      if is_scanned_block_param_path(path) and (config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False)):
        target_sharding = target_sharding.with_memory_kind("pinned_host")
      else:
        target_sharding = target_sharding.with_memory_kind("device")
    return target_sharding

  return jax.tree_util.tree_map_with_path(
      _update_sharding,
      state_mesh_shardings,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.sharding.Sharding)),
  )
```

---

### 4.3 Abstract State Annotation Contract: `set_named_sharding_nnx`

In [src/maxtext/utils/maxtext_utils_nnx.py](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/utils/maxtext_utils_nnx.py#L90-L155):
- Converts `abstract_state` leaves into `jax.ShapeDtypeStruct` annotated with the exact `NamedSharding` from `input_state_mesh_shardings`.
- Preserves the outer `nnx.Variable` container wrapper for NNX compatibility.

```python
def set_named_sharding_nnx(abstract_state: nnx.State, named_sharding: nnx.State) -> nnx.State:
  """Recursively tags abstract TrainStateNNX leaves with input mesh shardings and memory kinds."""
  def _set_leaf(x, y):
    sharding_val = getattr(y, "value", y)
    if hasattr(sharding_val, "sharding"):
      sharding_val = sharding_val.sharding
    if isinstance(x, nnx.Variable):
      val = x.get_value()
      shape = getattr(val, "shape", ())
      dtype = getattr(val, "dtype", jnp.float32)
      return x.replace(value=jax.ShapeDtypeStruct(shape, dtype, sharding=sharding_val))
    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding_val)

  return jax.tree.map(
      _set_leaf,
      abstract_state,
      named_sharding,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct, jax.sharding.Sharding)),
  )
```

---

### 4.4 Autodiff Gradient Staging, Optimizer Staging & Eviction Contract: `train_step`

In [src/maxtext/trainers/pre_train/train.py](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/trainers/pre_train/train.py#L570-L760):

#### Critical Gradient Memory Kind Invariant
> [!IMPORTANT]
> JAX Automatic Differentiation (`jax.value_and_grad`) derives the memory kind of output gradients (`raw_grads`) directly from the primal inputs (`state.model`). When scanned model parameters enter `train_step` in `pinned_host`, autodiff produces their corresponding gradient leaves in `pinned_host`.
>
> When `optax.tree_norm(raw_grads)` or `apply_gradient_clipping` computes the global gradient norm, JAX executes binary reductions (`lax.add`) across leaves. If one leaf is in `pinned_host` and another is in `device` HBM, JAX raises:
> `ValueError: memory_space of all inputs passed to add must be the same. Got one operand with type: float32<host> and another operand with type: float32`.
>
> **Mandatory Contract**: Immediately post-VJP, all gradient leaves must be constrained to `device` HBM before any norm or clipping arithmetic!

```python
# 1. Immediately post-VJP: Force all gradients to device HBM
def _force_grad_to_device(g):
  sharding = getattr(g, "sharding", None)
  if hasattr(sharding, "with_memory_kind"):
    return jax.lax.with_sharding_constraint(g, sharding.with_memory_kind("device"))
  return g

grads = jax.tree.map(_force_grad_to_device, raw_grads)

# 2. Gradient clipping executes purely on device HBM
if config.gradient_clipping_threshold > 0:
  grads = maxtext_utils.apply_gradient_clipping(grads, None, config.gradient_clipping_threshold)

# 3. Pre-Apply-Gradients: Stage scanned model parameters to device HBM
if config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False):
  model_params = nnx.state(state.model)
  
  def _stage_to_device(path, val):
    if not is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    s = _get_param_sharding(path, val)
    target_s = s.with_memory_kind("device")
    new_val = jax.lax.with_sharding_constraint(val_arr, target_s)
    if isinstance(val, nnx.Variable):
      return val.replace(value=new_val)
    return new_val

  dev_model_params = jax.tree_util.tree_map_with_path(
      _stage_to_device, model_params, is_leaf=_is_tree_leaf
  )
  nnx.update(state.model, dev_model_params)

# 4. Native Optax update on TPU hardware
if config.skip_step_on_spikes:
  grad_norm = max_utils.l2norm_pytree(grads)
  state.apply_gradients(grads, loss=loss, grad_norm=grad_norm)
else:
  state.apply_gradients(grads)

# 5. Post-Apply-Gradients: Evict scanned model parameters back to pinned_host
if config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False):
  model_params = nnx.state(state.model)
  
  def _evict_to_host(path, val):
    if not is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    s = _get_param_sharding(path, val)
    target_s = s.with_memory_kind("pinned_host")
    new_val = jax.lax.with_sharding_constraint(val_arr, target_s)
    if isinstance(val, nnx.Variable):
      return val.replace(value=new_val)
    return new_val

  host_model_params = jax.tree_util.tree_map_with_path(
      _evict_to_host, model_params, is_leaf=_is_tree_leaf
  )
  nnx.update(state.model, host_model_params)
```

#### Elimination of Invalid Nested JIT
All calls to nested `jax.jit(lambda a: ...)` or dynamic compilations within `train_step` are completely eliminated. `jax.lax.with_sharding_constraint` provides zero-overhead, purely symbolic HLO annotations that XLA translates directly into asynchronous `CopyH2D` and `CopyD2H` operations.

#### Buffer Donation Contract
In [get_functional_train_with_signature](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/utils/maxtext_utils.py#L93-L112):
```python
if getattr(config, "parameter_memory_two_layer_buffer", False) or config.parameter_memory_host_offload:
  donate_argnums = ()
else:
  donate_argnums = 0
```
`donate_argnums` is set to `()` (no buffer donation) when parameter host offload is active to prevent buffer aliasing conflicts between pinned host buffers and TPU device allocations across execution steps.

---

### 4.5 Scan Prefetch Contract: `nnx_scan.py`

In [src/maxtext/layers/nnx_scan.py](file:///usr/local/google/home/mohitkhatwani/maxtext_optoff/src/maxtext/layers/nnx_scan.py#L94-L158):

#### Slice Transfer and Memory Reclamation
When `jax.lax.scan` iterates over the stacked parameters `(params, state)`:
1. JAX automatically extracts the slice `current_params` for iteration $i$, reducing the rank by 1 (eliminating the leading scan axis).
2. Inside `scan_body`, each leaf of `current_params` is prefetched into `device` memory kind with matched rank PartitionSpec:
   ```python
   def scan_body(current_carry, scanned_state):
     current_params, current_state = scanned_state
     
     if parameter_memory_host_offload or parameter_memory_two_layer_buffer:
       def move_param_to_device(param):
         sharding = getattr(param, "sharding", None)
         if sharding is None and hasattr(param, "aval"):
           sharding = getattr(param.aval, "sharding", None)
         if hasattr(sharding, "with_memory_kind"):
           spec = getattr(sharding, "spec", None)
           ndim = getattr(param, "ndim", len(param.shape) if hasattr(param, "shape") else None)
           if spec is not None and ndim is not None and len(spec) == ndim + 1:
             # Drop leading scan axis partition for the unstacked slice
             target_spec = jax.sharding.PartitionSpec(*spec[1:])
             target_sharding = jax.sharding.NamedSharding(sharding.mesh, target_spec, memory_kind="device")
           else:
             target_sharding = sharding.with_memory_kind("device")
           return jax.lax.with_sharding_constraint(param, target_sharding)
         return max_utils.to_device(param)

       current_params = jax.tree.map(move_param_to_device, current_params)
       
     current_layer = nnx.merge(layer_graphdef, current_params, current_state)
     next_carry = apply_fn(current_layer, current_carry)
     
     # CRITICAL: Exclude nnx.Param from returned state to allow immediate HBM reclamation
     non_param_state = nnx.state(current_layer, nnx.Not(nnx.Param))
     return next_carry, non_param_state
   ```

3. **Ephemeral Guarantee**: Because `current_params` is never returned as part of the scan carry or scan outputs, its device HBM buffer is strictly scoped to the iteration body and is immediately freed/reused by XLA for subsequent layer prefetches.

---

## 5. XLA Compiler, HLO Lowering & Hardware Overlap Mechanics

### 5.1 HLO Op Code Sequences
When lowered through XLA, the memory kind annotations compile into asynchronous DMA streams:
1. **Forward Prefetch Loop**:
   - `custom-call @HostToDeviceEnclosingCopy`: Asynchronous transfer of slice $i+1$ initiated while MXU computes slice $i$.
2. **Post-VJP Gradient Alignment**:
   - Ensures gradients computed by backward VJP are uniformly in `device` HBM.
3. **Pre-Apply-Gradients Transfer**:
   - `custom-call @HostToDeviceEnclosingCopy`: Staging full stacked parameter tensor from pinned host DRAM to TPU HBM.
4. **Optax Compute**:
   - `fusion` containing AdamW / Muon update arithmetic executed directly on TPU vector/matrix registers.
5. **Post-Apply-Gradients Transfer**:
   - `custom-call @DeviceToHostEnclosingCopy`: Eviction of full updated stacked parameter tensor from TPU HBM to pinned host DRAM.

### 5.2 XLA Scheduling Flags
To achieve maximum communication-computation overlap without pipeline stalls, the following compiler options are enabled via `LIBTPU_INIT_ARGS`:
- `--xla_tpu_enable_latency_hiding_scheduler=true`: Enables ILP-based latency hiding scheduler.
- `--xla_tpu_host_transfer_overlap_limit=4`: Allows up to 4 concurrent host-device transfers in flight.
- `--xla_max_concurrent_host_send_recv=100`: Permits high-concurrency host-device DMA queues.
- `--xla_lhs_prioritize_async_depth_over_stall=ENABLED`: Prioritizes issuing prefetches ahead of compute stalls.
- `--xla_should_allow_loop_variant_parameter_in_chain=ENABLED`: Permits loop-carried parameter streaming across scan iterations.

---

## 6. Comprehensive Verification and Invariant Validation

### 6.1 Invariant Verification Checklist

| Invariant ID | Verification Criterion | Expected State | Validation Method |
|---|---|---|---|
| **INV-1** | Input Sharding Equivalence | `abstract_state` leaf shardings match `input_state_mesh_shardings` | `test_sharding_match.py` passes with 0 metadata mismatches. |
| **INV-2** | Scanned Parameter Isolation | ONLY `local_layers` under `scanned_blocks` have `memory_kind="pinned_host"` | Inspect `input_state_mesh_shardings` leaf sharding metadata. |
| **INV-3** | Optimizer State Protection | `opt_state.mu`, `opt_state.nu`, `step`, `count` have `memory_kind="device"` | Inspect `TrainStateNNX.optimizer` sharding metadata; verify no `pinned_host` tags. |
| **INV-4** | Non-Scanned Param Protection | Embeddings, norms, global layers, and remainder layers have `memory_kind="device"` | Inspect `TrainStateNNX.model` non-scanned paths. |
| **INV-5** | Gradient Device Uniformity | All leaves in `raw_grads` constrained to `device` before `tree_norm` / clipping | `apply_gradient_clipping` passes without `ValueError: memory_space`. |
| **INV-6** | Closed-Loop Idempotency | `out_shardings` of `train_step` matches `in_shardings` exactly | Verify JIT output signature in `get_functional_train_with_signature`. |
| **INV-7** | Zero Nested JIT | No `jax.jit` calls inside `train_step` or `scan_body` | Static AST / code audit. |
| **INV-8** | Ephemeral Scan Buffers | No `nnx.Param` in `scan_body` return tuple | Verify `nnx_scan.py` returns `(next_carry, non_param_state)`. |
| **INV-9** | AOT Compilation Success | Full AOT compilation of 80B model succeeds on v6e-256 | `train_compile` completes HLO lowering and dumps `/tmp/hlo_dump.hlo`. |
