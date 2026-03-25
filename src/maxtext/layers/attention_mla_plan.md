# Kernel Optimization Plan: MLA Indexer Computation

## 1. Current Kernel Analysis
The current implementation of the MLA Indexer computation involves three Pallas kernels:
1.  **Forward Kernel (`Indexer.kernel`)**: Computes attention scores using a shared Key (MQA-style) and weighted head aggregation.
2.  **Backward Kernel 1 (`backward_qw_kernel`)**: Computes gradients for Query (`d_q`) and Head Weights (`d_w`).
3.  **Backward Kernel 2 (`backward_k_kernel`)**: Computes gradients for Key (`d_k`).

**Identified Issues:**
-   **Serialized Execution**: All kernels currently use a "start DMA -> wait DMA -> compute" pattern within their inner loops. This prevents overlap of memory transfer and computation, significantly reducing performance on TPU where HBM bandwidth is often the bottleneck.
-   **Single Buffering**: Scratch buffers in VMEM are single-buffered, making it impossible to prefetch the next block while processing the current one.
-   **Block Sizing**: `bS=256` and `bT=32` are hardcoded. While reasonable, they should be validated against the specific head dimensions and VMEM capacity.

## 2. Optimization Strategy
The primary optimization is to implement **Manual Software Pipelining (Double Buffering)** for all three kernels.

**Key Transformations:**
1.  **Double Buffering**: Allocate scratch buffers of size `(2, ...)` in VMEM for all inputs that are iterated over (e.g., `K` blocks in forward pass).
2.  **Pipelined Loop Structure**:
    -   **Prologue**: Initiate the load for the first block (buffer 0).
    -   **Body**:
        -   Wait for buffer `i % 2`.
        -   Initiate load for block `i+1` into buffer `(i+1) % 2` (if not last iteration).
        -   Compute using buffer `i % 2`.
    -   **Epilogue**: (Handled naturally by the loop condition).
3.  **Async Copies**: Use `pltpu.make_async_copy` with explicit semaphores to manage synchronization.

## 3. Memory Layout and Tiling

### Forward Kernel (`Indexer.kernel`)
-   **Grid**: `(B, T // bT)`
-   **Loop**: Over `S // bS` blocks.
-   **Stationary Data**: `q_block` (bT, H, D), `w_block` (bT, H) - Loaded once per program, stay in VMEM.
-   **Streaming Data**: `k_block` (bS, D), `mask_block` (bT, bS).
-   **Scratch Buffers**:
    -   `k_scratch`: `(2, bS, D_padded)` in VMEM.
    -   `mask_scratch`: `(2, bT, bS)` in VMEM.
    -   `score_scratch`: `(bT, bS)` in VMEM (Accumulator, no need to double buffer if we write out once).

### Backward Kernel 1 (`backward_qw_kernel`)
-   **Grid**: `(B, T // bT)`
-   **Loop**: Over `S // bS` blocks.
-   **Stationary Data**: `q_block`, `w_block` (loaded once). `d_q_acc`, `d_w_acc` (accumulators in VMEM).
-   **Streaming Data**: `k_block`, `d_score_block`.
-   **Scratch Buffers**:
    -   `k_scratch`: `(2, bS, D_padded)`
    -   `d_score_scratch`: `(2, bT, bS)`

### Backward Kernel 2 (`backward_k_kernel`)
-   **Grid**: `(B, S // bS)`
-   **Loop**: Over `T // bT` blocks.
-   **Stationary Data**: `k_block` (loaded once). `d_k_acc` (accumulator).
-   **Streaming Data**: `q_block`, `w_block`, `d_score_block`.
-   **Scratch Buffers**:
    -   `q_scratch`: `(2, bT, H, D_padded)`
    -   `w_scratch`: `(2, bT, H_padded)`
    -   `d_score_scratch`: `(2, bT, bS)`

## 4. TPU-Specific Optimizations
-   **Vector Alignment**: Ensure `D` and `H` are padded to multiples of 128 (already partially handled, will reinforce).
-   **Semaphores**: Use `pltpu.SemaphoreType.DMA` for async copy tracking.
-   **Predication**: Use `pl.when` to handle the conditional prefetch for the next iteration.

## 5. Implementation Details

### Pipeline Logic (Template)
```python
# Example for Forward Kernel Loop
def body(i, _):
    curr_buff = i % 2
    next_buff = (i + 1) % 2
    
    # 1. Wait for current block
    # (In first iteration, this waits for the copy started in prologue)
    # (In subsequent, it waits for copy started in previous body)
    # We need a semaphore per buffer to track "ready to read"
    
    # Actually, simpler pattern:
    # Start 0.
    # Loop i:
    #   Wait i%2.
    #   Start (i+1)%2 if not last.
    #   Compute i%2.
```

### Block Sizes
-   `bT = 32`: Good balance for register pressure and T-dimension parallelism.
-   `bS = 128`: Reduced from 256 to ensure double buffering fits comfortably in VMEM with larger head dimensions.
    -   Check: `2 * 128 * 256 * 4 bytes` = ~256KB. Very small. We can keep `bS=256` or even `512`.
    -   Let's stick to `bS=256` (approx 512KB for double buffer).

## 6. Expected Performance Impact
-   **Latency**: Significant reduction due to hiding HBM latency.
-   **Throughput**: Higher utilization of MXU (Matrix Units) as they won't stall waiting for data.
-   **Speedup**: Estimated 1.5x - 2.0x improvement for memory-bound regimes.

## 7. Documentation Requirements
-   Annotate all scratch buffer shapes with `(2, ...)` to indicate double buffering.
-   Clearly comment the "Produce / Consume" pattern in the pipeline.
-   Document the memory hierarchy (HBM -> VMEM -> Registers).
