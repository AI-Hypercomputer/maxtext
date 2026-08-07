# Milestone: Double-Buffered Layer-Wise Parameter & Optimizer State Host Offloading for Scanned MaxText Models

## User Requirements & Scope Constraints
- **Primary Target**: Scanned Transformer decoder models (`param_scan_axis=0` / `scan_layers=True`).
- **First Priority**: Parameter memory host offloading (`parameter_memory_host_offload=True`), followed by Optimizer State offloading (`optimizer_memory_host_offload=True`).
- **Pipelining Strategy**: Double-buffered software-pipelined host<->device streaming integrated directly into the scanned layer execution loop (`jax.lax.scan`).

## Machine-Checkable 'Done' Criteria

1. **True HBM Offloading for Scanned Parameters & Opt State**:
   - Parameter tensors and optimizer state tensors stacked along `param_scan_axis=0` are maintained in CPU host memory (`pinned_host`).
   - HBM is allocated strictly for active layer slices (layer $i$ and pre-fetched layer $i+1$) rather than loading all layer weights/states into HBM simultaneously.

2. **Double-Buffered Pre-fetching in Scanned Loop**:
   - Host-to-device transfer of layer $i+1$ parameters/optimizer state is initiated concurrently with layer $i$ compute within `jax.lax.scan`.
   - Updated parameters and optimizer state for layer $i$ are transferred back to host memory.

3. **Mathematical Correctness & Equivalence**:
   - Training loss and gradient trajectories match non-offloaded baseline up to numerical precision.

4. **Unit & Integration Testing**:
   - Unit tests pass with scanned offloading enabled.
