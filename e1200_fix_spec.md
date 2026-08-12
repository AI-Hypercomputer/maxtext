# Architectural Design Specification: Eliminating E1200 Output Location Mismatch and Establishing Strict Parameter Host Offloading Contracts in MaxText NNX

**Document Version:** 1.0  
**Target Hardware:** Cloud TPU v6e-256 (32 GiB HBM per chip)  
**Target Model:** Qwen3-Next-80B-A3B (MoE Hybrid GDN, 80B total / 3B active params)  
**Target Branch:** `mohit/bharat_opt_off`  
**Author:** Principal JAX & MaxText Distributed Systems Architect  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Context and Encountered Failure
During Ahead-Of-Time (AOT) compiler lowering and compilation of `qwen3-next-80b-a3b` on a 256-chip TPU v6e topology (`hybrid_ring_64x4` mesh) with Layer-Wise Parameter Host Offloading enabled (`parameter_memory_host_offload=True`, `parameter_memory_two_layer_buffer=True`, `optimizer_memory_host_offload=False`), the XLA compiler aborts during the backend `HostOffloader` pass with the following fatal error:

```text
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: E1200: CompileTimeHostOffloadOutputLocationMismatch:
Tensor which is moved to host (starting from square.790) is returned from the entry computation but the layout for this output is not set to host memory.
See https://openxla.org/xla/errors/error_1200 for more details.
```

### 1.2 Core Architectural Invariant
To run large-scale models like Qwen3-80B on memory-constrained topologies with large per-device batch sizes (e.g., `pdb=16`), MaxText requires:
1. **Scanned Block Parameters** (`state.model` decoder layers): Must reside in `pinned_host` memory space in `in_shardings` and `out_shardings`. During forward/backward execution, per-layer parameter slices are streamed dynamically into TPU High Bandwidth Memory (HBM, `device` memory kind).
2. **Optimizer States** (`state.optimizer.opt_state`): Must reside **strictly and permanently on `device`** (HBM) in both `in_shardings` and `out_shardings`. No optimizer state tensor (`mu`, `nu`, `step`, `count`) may ever be offloaded to host memory or tagged with `pinned_host`.
3. **Non-Scanned Model Parameters** (`state.model` embedding table, final RMSNorm, unembedding head, and any unscanned remainder layers): Must reside **strictly and permanently on `device`** in both `in_shardings` and `out_shardings`.
4. **Parameter Update & Eviction Lifecycle**: Gradient calculations and optimizer updates (`AdamW`) run purely in TPU HBM. The updated model weights are explicitly evicted back to `pinned_host` using explicit sharding constraints before the computation exits, guaranteeing that optimizer states remain in device HBM without compiler layout conflicts.

---

## 2. Deep Root Cause & Dataflow Analysis of Error E1200

### 2.1 The Origin of `square.790`
In Optax AdamW (`optax.scale_by_adam` / `optax.adamw`), the second-moment accumulator $\nu$ is updated at each training step according to:
$$\nu_t = \beta_2 \nu_{t-1} + (1 - \beta_2) \cdot g_t^2$$
In XLA HLO, the term $g_t^2$ (the element-wise square of the gradient) is lowered to an HLO instruction such as `%square.790 = chlo.square %grad` (or `%integer_pow %grad, y=2`).

The computed second moment $\nu_t$ has two downstream consumers:
1. **The Entry Computation Output for Optimizer State:** $\nu_t$ is stored directly in `state.optimizer.opt_state` and returned as part of the `train_step` output state PyTree. In `out_shardings`, this tensor is annotated with `NamedSharding(..., memory_kind="device")`.
2. **The Parameter Update Pipeline:** $\nu_t$ is consumed to compute parameter updates $\Delta \theta_t = -\eta \cdot \left(\frac{\hat{\mu}_t}{\sqrt{\hat{\nu}_t} + \epsilon} + \lambda \theta_t\right)$. The updated model parameters $\theta_{t+1} = \theta_t + \Delta \theta_t$ are then placed into `state.model`.

### 2.2 Why XLA's `HostOffloader` Detected an Output Location Mismatch
The XLA TPU compiler's `HostOffloader` pass traces dataflow graphs containing host-offload instructions (`custom-call @DeviceToHostEnclosingCopy`, `device_put(pinned_host)`, or memory space annotations). When `HostOffloader` executes:

1. **Autodiff VJP Backward Propagation:** In `train.py`, `curr_params` was passed to `jax.value_and_grad(diff_wrapper)`. Because `curr_params` entered `train_step` with `pinned_host` sharding, the primal inputs were typed as `float32<host>`. As a result, JAX's reverse-mode automatic differentiation generated gradients (`raw_grads`) whose abstract values were associated with `host` memory space.
2. **Tracer Sharding Stripping:** Inside `jax.jit(train_step)`, JAX represents array values as `DynamicJaxprTracer` objects. Tracers do not have a `.sharding` attribute (`getattr(tracer, "sharding", None)` evaluates to `None`). Consequently:
   - In `train.py`, `_force_grad_to_device` checked `getattr(g, "sharding", None)`, which returned `None`, falling back to `max_utils.to_device(g)`. While `max_utils.to_device` called `jax.device_put(g, Space.Device)`, it did not apply the full `NamedSharding(mesh, PartitionSpec, memory_kind="device")`.
   - In `train_state_nnx.py`, `TrainStateNNX.apply_gradients` executed `_to_device_var`, which similarly evaluated `getattr(arr, "sharding", None)` as `None`, silently becoming a no-op during tracing.
3. **Graph Clustering and Host-Sinking:** In `train.py` Step 5 (`_evict_to_host`), the updated model parameters $\theta_{t+1}$ (which depend on $\Delta \theta_t$, which depends on $\nu_t$, which originates from `square.790`) were explicitly moved to host memory via `jax.device_put(val_arr, host_sharding)`. 
4. **Dual-Destination Conflict:** Because `square.790` fed into the host-evicted $\theta_{t+1}$, the `HostOffloader` compiler pass clustered the upstream optimizer update graph and marked the intermediate calculations (including `square.790` and $\nu_t$) as host-offloaded computations. However, $\nu_t$ is simultaneously returned directly in `out_shardings` as part of `state.optimizer.opt_state` with `memory_kind="device"`.
5. **Direct Jaxpr Trace Proof:** Tracing the JAXPR equations of `train_step` directly revealed the concrete manifestation of this error:
   - Equations 4132–4285 and 4403–4625 execute `a:f32 = square b` in **device** memory space (corresponding to gradient squaring in AdamW `tx.update`).
   - However, equations 4628–4736 execute `a:f32<host>[13,2,64] = square b` in **host** memory space (corresponding to `l2norm_pytree(model_params)` executed on `new_state.model` **after** host eviction).
   - Because `square` operations existed in both `f32` (device) and `f32<host>` (host) across the shared parameter/optimizer lifecycle, XLA's `HostOffloader` pass attempted to offload the intermediate nodes (including `square.790` / `square.1129`) to host memory, conflicting with the entry output contract for optimizer moments in device memory kind and triggering `E1200: CompileTimeHostOffloadOutputLocationMismatch`.

---

## 3. PyTree Sharding & Memory Kind Contracts

To eliminate E1200 errors and enforce complete memory isolation, the system must adhere to strict mathematical sharding contracts across all PyTree components.

```
+---------------------------------------------------------------------------------------------------+
|                                      TrainStateNNX PyTree                                         |
+-------------------------------------------------+-------------------------------------------------+
|               state.model                       |               state.optimizer                   |
+------------------------+------------------------+------------------------+------------------------+
| Scanned Block Params   | Non-Scanned Params     | Optimizer Moments      | Step & Counts          |
| (local_layers.*)       | (embed, head, norm)    | (mu, nu in opt_state)  | (step, count)          |
+------------------------+------------------------+------------------------+------------------------+
| In:  pinned_host       | In:  device            | In:  device            | In:  device            |
| Out: pinned_host       | Out: device            | Out: device            | Out: device            |
+------------------------+------------------------+------------------------+------------------------+
```

### 3.1 Contract Definitions

#### A. Scanned Block Parameters (`state.model`)
- **Applies to:** Any parameter path containing `scanned_blocks` or `local_layers` (e.g., `decoder/scanned_blocks/local_layers/attention/kernel`).
- **Excludes:** Non-scanned embedding tables, output projection heads, and global root layer norms.
- **Contract:**
  $$\text{in\_sharding}(\theta_{\text{scanned}}) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"pinned\_host"})$$
  $$\text{out\_sharding}(\theta_{\text{scanned}}) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"pinned\_host"})$$

#### B. Non-Scanned Model Parameters (`state.model`)
- **Applies to:** Embedding layers (`decoder/embedding/kernel`), output heads (`decoder/logits_dense/kernel`), and root normalization layers.
- **Contract:**
  $$\text{in\_sharding}(\theta_{\text{unscanned}}) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"device"})$$
  $$\text{out\_sharding}(\theta_{\text{unscanned}}) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"device"})$$

#### C. Optimizer States (`state.optimizer`)
- **Applies to:** All leaves within `state.optimizer.opt_state` (first moment $\mu$, second moment $\nu$, EMA states) and scalar counters (`state.optimizer.step`).
- **Contract:**
  $$\forall s \in \text{Leaves}(\text{state.optimizer}), \quad \text{memory\_kind}(s) = \text{"device"}$$
  $$\text{in\_sharding}(s) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"device"})$$
  $$\text{out\_sharding}(s) = \text{NamedSharding}(\text{mesh}, \text{PartitionSpec}(\dots), \text{memory\_kind}=\text{"device"})$$

---

## 4. End-to-End Step Lifecycle & Staging/Eviction Mechanics

The complete execution lifecycle within `train_step` is partitioned into six non-overlapping phases:

```
[1. Input Binding] 
       │  (scanned params in pinned_host, optimizer states in device HBM)
       ▼
[2. Forward / Backward (VJP)]
       │  (nnx_scan prefetches per-layer slices into device HBM; VJP runs)
       ▼
[3. Immediate Post-VJP Gradient Normalization]
       │  (grads mapped to device HBM with explicit PartitionSpecs)
       ▼
[4. Pre-Apply-Gradients Model Staging]
       │  (state.model scanned params staged to device HBM via device_put)
       ▼
[5. Optax apply_gradients Execution]
       │  (pure HBM execution: new_params, mu, nu computed strictly on TPU)
       ▼
[6. Post-Apply-Gradients Eviction]
       │  (state.model scanned params evicted to pinned_host; opt_state untouched)
       ▼
[Output Return]
       (Matches out_shardings: scanned params in pinned_host, optimizer in device)
```

### Phase 1: Input Binding & Sharding Boundary
- `input_state_mesh_shardings` is constructed by `sharding.build_zero1_input_state_mesh_shardings`.
- `in_shardings` and `out_shardings` are passed to `jax.jit(train_step)` at the top level.
- Scanned parameter leaves enter the computation with `memory_kind="pinned_host"`. All optimizer state leaves enter with `memory_kind="device"`.

### Phase 2: Forward & Backward Passes (Layer-Wise Prefetch)
- In `src/maxtext/layers/nnx_scan.py`, `apply_scanned_layers` executes `jax.lax.scan`.
- Inside `scan_body`, each layer's parameter slice is prefetched to TPU HBM via:
  ```python
  target_sharding = jax.sharding.NamedSharding(sharding.mesh, target_spec, memory_kind="device")
  current_params = jax.lax.with_sharding_constraint(param, target_sharding)
  ```
- The forward activations and backward adjoints are computed layer-by-layer, allowing parameters to be streamed through TPU HBM without materializing the full model parameter set on device at once.

### Phase 3: Immediate Post-VJP Gradient Device Binding
- Immediately after `jax.value_and_grad` completes, the gradients `raw_grads` (which may carry host annotations inherited from the primal inputs) are rebound to device HBM using their canonical `PartitionSpec`:
  ```python
  def _bind_grad_to_device(path, grad_arr):
    s = _lookup_canonical_sharding(path)
    target_sharding = s.with_memory_kind("device") if hasattr(s, "with_memory_kind") else s
    return jax.device_put(grad_arr, target_sharding)
  
  grads = jax.tree_util.tree_map_with_path(_bind_grad_to_device, raw_grads)
  ```

### Phase 4: Pre-Apply-Gradients Model Staging
- Before updating parameters, the scanned parameters in `state.model` are staged to TPU HBM:
  ```python
  def _stage_to_device(path, val):
    if not sharding.is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    s = _lookup_canonical_sharding(path)
    target_s = s.with_memory_kind("device")
    new_val = jax.device_put(val_arr, target_s)
    return val.replace(value=new_val) if isinstance(val, nnx.Variable) else new_val

  dev_model_params = jax.tree_util.tree_map_with_path(_stage_to_device, nnx.state(state.model))
  nnx.update(state.model, dev_model_params)
  ```

### Phase 5: Hardware-Accelerated Optax Gradient Application
- `state.apply_gradients(grads)` is executed on `TrainStateNNX`.
- Inside `nnx.Optimizer.update`:
  - `param_arrays` (all in device HBM), `grad_arrays` (all in device HBM), and `opt_state_arrays` (all in device HBM) are passed to `self.tx.update`.
  - AdamW computes `new_opt_state` ($\mu_t, \nu_t$) and `updates` $\Delta \theta_t$ entirely in TPU HBM.
  - `new_params` = $\theta_t + \Delta \theta_t$ is computed in TPU HBM.
  - `nnx.update(self.model, new_params)` updates model weights in place.
  - `nnx.update(self.opt_state, nnx.state(new_opt_state))` updates optimizer moments in place.
- **Zero Host Interaction:** During this phase, not a single tensor is placed on or transferred to host memory.

### Phase 6: Post-Apply-Gradients Parameter Eviction
- Only `state.model` scanned parameters are evicted back to `pinned_host`. `state.optimizer` is completely bypassed:
  ```python
  def _evict_to_host(path, val):
    if not sharding.is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    s = _lookup_canonical_sharding(path)
    target_s = s.with_memory_kind("pinned_host")
    new_val = jax.device_put(val_arr, target_s)
    return val.replace(value=new_val) if isinstance(val, nnx.Variable) else new_val

  host_model_params = jax.tree_util.tree_map_with_path(_evict_to_host, nnx.state(state.model))
  nnx.update(state.model, host_model_params)
  ```
- **Metric Computation Placement:** Metric calculations (such as `l2norm_pytree(grads)`) are performed using `grads` (in device HBM) and parameter norms are evaluated on staged parameters before eviction, preventing host tensor math.

---

## 5. Activation Offloading & Batch Size Coordination (`pdb=16`)

### 5.1 Memory Budget on TPU v6e-256 (32 GiB HBM / Device)
When training Qwen3-Next-80B-A3B with `pdb=16` (per-device batch size 16), activation memory pressure increases substantially:

| Memory Component | Allocation without Host Offload | Allocation with Param Offload | Allocation with Param + Act Offload |
| :--- | :--- | :--- | :--- |
| **Model Parameters (Scanned)** | ~12.5 GiB (HBM) | **0.4 GiB** (2-layer buffer in HBM) | **0.4 GiB** (HBM) |
| **Optimizer States ($\mu, \nu$)** | ~8.0 GiB (HBM) | ~8.0 GiB (HBM) | ~8.0 GiB (HBM) |
| **Model Parameters (Host)** | 0 GiB | **~12.5 GiB** (Host DRAM) | **~12.5 GiB** (Host DRAM) |
| **Activations (`pdb=16`, `seq=2048`)** | ~18.0 GiB (HBM, OOM) | ~18.0 GiB (HBM, Tight) | **~6.5 GiB** (HBM, Remainder on Host) |
| **Workspace & Temporary Buffers** | ~4.0 GiB (HBM) | ~4.0 GiB (HBM) | ~4.0 GiB (HBM) |
| **Total Peak HBM Footprint** | **> 42.5 GiB (OOM)** | **~30.4 GiB (Near Limit)** | **~18.9 GiB (Safe & Optimal)** |

### 5.2 Offloading Coordination Flags
To guarantee stable execution without PCIe / ICI bandwidth saturation:
1. `parameter_memory_host_offload=True` & `parameter_memory_two_layer_buffer=True`: Enables 2-layer ping-pong double buffering for scanned block parameters.
2. `decoder_layer_input=offload`: Offloads inter-layer residual activations to host memory between transformer layers, streaming them back before layer execution.
3. `context=offload`: Offloads sequence attention context states during long-sequence processing.
4. `optimizer_memory_host_offload=False`: Keeps optimizer states permanently in HBM, ensuring maximum AdamW compute throughput.

---

## 6. Concrete Code Implementations

### 6.1 Updates to `src/maxtext/trainers/pre_train/train.py`

#### A. Canonical Sharding Lookup Helper
Tracers do not expose `.sharding`. Shardings must be looked up from `params_shardings` or `state_mesh_shardings` by normalized path:

```python
def _build_sharding_lookup(mesh_shardings):
  lookup = {}
  for path, s in jax.tree_util.tree_flatten_with_path(mesh_shardings)[0]:
    norm_path = tuple(
        str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        for k in path
        if str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        not in ["model", "optimizer", "opt_state", "value"]
    )
    if norm_path not in lookup and isinstance(s, jax.sharding.Sharding):
      lookup[norm_path] = s
  return lookup
```

#### B. Gradient Device Normalization (Post-VJP)
```python
# 1. Immediately post-VJP: Force all gradients to device HBM with correct NamedSharding
sh_lookup = _build_sharding_lookup(state_mesh_shardings)

def _force_grad_to_device(path, g):
  norm_p = tuple(
      str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
      for k in path
      if str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
      not in ["model", "optimizer", "opt_state", "value"]
  )
  s = sh_lookup.get(norm_p, None)
  if s is not None and hasattr(s, "with_memory_kind"):
    target_s = s.with_memory_kind("device")
    return jax.device_put(g, target_s)
  return max_utils.to_device(g)

grads = jax.tree_util.tree_map_with_path(
    _force_grad_to_device,
    raw_grads,
    is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct)) or (hasattr(x, "shape") and hasattr(x, "dtype")),
)
```

#### C. Pre-Apply-Gradients Model Staging & Post-Apply-Gradients Parameter Eviction
```python
# 3. Pre-Apply-Gradients: Stage scanned model parameters to device HBM
if config.parameter_memory_host_offload or getattr(config, "parameter_memory_two_layer_buffer", False):
  model_params = nnx.state(state.model)

  def _stage_to_device(path, val):
    if not sharding.is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    norm_p = tuple(
        str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        for k in path
        if str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        not in ["model", "optimizer", "opt_state", "value"]
    )
    s = sh_lookup.get(norm_p, None)
    if s is not None and hasattr(s, "with_memory_kind"):
      target_s = s.with_memory_kind("device")
      new_val = jax.device_put(val_arr, target_s)
    else:
      new_val = max_utils.to_device(val_arr)
    if isinstance(val, nnx.Variable):
      return val.replace(value=new_val)
    return new_val

  dev_model_params = jax.tree_util.tree_map_with_path(
      _stage_to_device,
      model_params,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct)) or (hasattr(x, "shape") and hasattr(x, "dtype")),
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
    if not sharding.is_scanned_block_param_path(path):
      return val
    val_arr = getattr(val, "value", val)
    norm_p = tuple(
        str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        for k in path
        if str(k.key if hasattr(k, "key") else k.name if hasattr(k, "name") else k)
        not in ["model", "optimizer", "opt_state", "value"]
    )
    s = sh_lookup.get(norm_p, None)
    if s is not None and hasattr(s, "with_memory_kind"):
      target_s = s.with_memory_kind("pinned_host")
      new_val = jax.device_put(val_arr, target_s)
    else:
      new_val = val_arr
    if isinstance(val, nnx.Variable):
      return val.replace(value=new_val)
    return new_val

  host_model_params = jax.tree_util.tree_map_with_path(
      _evict_to_host,
      model_params,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct)) or (hasattr(x, "shape") and hasattr(x, "dtype")),
  )
  nnx.update(state.model, host_model_params)
```

### 6.2 Updates to `src/maxtext/common/train_state_nnx.py`

Clean up `TrainStateNNX.apply_gradients` to delegate staging to `train.py` and prevent redundant tracer checks:

```python
def apply_gradients(self, grads: Any, **kwargs):
  """Applies gradients using the internal optimizer.
  
  Parameters and gradients must be staged to device HBM prior to this call.
  All optimizer state updates occur purely in TPU HBM.
  """
  if self.optimizer is None:
    raise RuntimeError(f"Optimizer is not initialized on TrainStateNNX {self!r}")
  self.optimizer.update(self.model, grads, **kwargs)
```

---

## 7. Verification & Acceptance Criteria

### 7.1 AOT Compilation Verification
Run Ahead-Of-Time compilation of `qwen3-next-80b-a3b` on `v6e-256`:
```bash
/usr/local/google/home/mohitkhatwani/max_venv/bin/python -m maxtext.trainers.pre_train.train_compile \
  src/maxtext/configs/base.yml \
  model_name=qwen3-next-80b-a3b \
  compile_topology=v6e-256 \
  compile_topology_num_slices=1 \
  dataset_type=synthetic \
  dataset_name=synthetic \
  dtype=bfloat16 \
  allow_split_physical_axes=True \
  ici_expert_parallelism=4 \
  use_ring_of_experts=True \
  custom_mesh=hybrid_ring_64x4 \
  use_ragged_sort=True \
  use_random_routing=True \
  num_moe_token_chunks=2 \
  per_device_batch_size=8 \
  opt_type=adamw \
  max_target_length=2048 \
  ragged_buffer_factor=1.5 \
  remat_policy=custom \
  reuse_example_batch=1 \
  decoder_layer_input=device \
  ici_fsdp_parallelism=-1 \
  steps=15 \
  hardware=tpu \
  sparse_matmul=True \
  megablox=True \
  optimizer_memory_host_offload=False \
  parameter_memory_host_offload=True \
  parameter_memory_two_layer_buffer=True \
  param_scan_axis=0 \
  enable_checkpointing=False \
  async_checkpointing=False \
  tokenizer_type=tiktoken \
  tokenizer_path=tokenizer_74B/ \
  use_gdn_kernel=True \
  use_hybrid_gdn=True
```

**Success Criteria:**
1. JIT lowering completes with 0 errors.
2. XLA backend compilation finishes with `0` `CompileTimeHostOffloadOutputLocationMismatch` (E1200) errors.
3. HLO module verification confirms that:
   - Scanned block parameters have `memory_space=1` (`pinned_host`) in entry inputs and outputs.
   - Optimizer states (`mu`, `nu`, `step`) have `memory_space=0` (`device`) in entry inputs and outputs.
   - All AdamW second-moment calculations (`chlo.square` / `integer_pow`) execute purely in device memory space without host clustering.
