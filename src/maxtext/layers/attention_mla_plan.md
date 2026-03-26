# Kernel Optimization Plan: MLA Indexer

## 1. Current Kernel Analysis
The current implementation of the MLA Indexer kernels (forward and backward) uses basic Pallas features but lacks advanced optimization techniques like pipelining and optimal block sizing.

**Current approach:**
- **Forward:** Loads `K` and `Mask` blocks from HBM inside the loop. Computes scores and writes them back to HBM. Uses `jax.lax.fori_loop`.
- **Backward:** Two kernels (`backward_qw`, `backward_k`). Both load inputs from HBM inside loops and accumulate gradients in VMEM.
- **Block Sizes:** `bT=32`, `bS=256`. These are relatively small for TPU v5p, potentially leading to low occupancy and high overhead.
- **Bottlenecks:**
    - **Memory Latency:** No overlap between computation and memory transfers (HBM <-> VMEM). The kernels wait for DMAs to complete before starting computation.
    - **Small Tiles:** Small block sizes may not fully utilize the Matrix Multiply Unit (MXU) and Vector Processing Unit (VPU).

## 2. Optimization Strategy
We will optimize the kernels by implementing **software pipelining** (double buffering) and increasing **block sizes**.

**Key Transformations:**
1.  **Pipelining:** Use double buffering for inputs loaded from HBM (`K`, `Mask`, `Q`, `W`, `d_Score`). This allows loading the next block while computing the current block, hiding memory latency.
2.  **Block Size Tuning:** Increase `bT` and `bS` to better amortize overheads and utilize TPU resources.
3.  **Vectorization:** Ensure all dimensions involved in DMA and dense computations are multiples of 128 (TPU vector size).

## 3. Memory Layout and Tiling
**Proposed Block Sizes:**
- **Forward Kernel:**
    - `bT = 128` (was 32)
    - `bS = 1024` (was 256)
    - **Rationale:** `bT=128` aligns with typical TPU block sizes. `bS=1024` provides a large enough reduction dimension.
    - **VMEM Usage:**
        - `Q`: `128 * 8 * 128 * 4B` ≈ 0.5 MB
        - `W`: `128 * 8 * 4B` ≈ 4 KB
        - `K` (2 buffers): `2 * 1024 * 128 * 4B` ≈ 1 MB
        - `Mask` (2 buffers): `2 * 128 * 1024 * 4B` ≈ 1 MB
        - `Score` (2 buffers): `2 * 128 * 1024 * 4B` ≈ 1 MB
        - **Total:** ~3.5 MB, well within TPU v5p VMEM capacity (16MB+).

- **Backward Kernels:**
    - `bT = 128`, `bS = 1024` (consistent with forward).

**Memory Layout:**
- **Inputs:** `Q`, `W`, `K`, `Mask`, `d_Score` in HBM (ANY).
- **Outputs:** `Score`, `d_Q`, `d_W`, `d_K` in HBM (ANY).
- **Scratch:** All intermediate buffers in VMEM.

## 4. TPU-Specific Optimizations
- **Manual Pipelining:** We will use `pltpu.make_async_copy` with semaphores to implement a manual double-buffering loop.
    - **Prologue:** Start loading block 0.
    - **Loop:**
        - Wait for block `i` load.
        - Start loading block `i+1`.
        - Compute on block `i`.
        - Start writing result `i` (for forward kernel).
- **Vector Alignment:** Pad `D` to 128 (already done). Pad `H` in `W` if necessary.

## 5. Implementation Details

### Forward Kernel (`kernel`)
- **Grid:** `(B, T // bT)`
- **Loop:** Iterate over `S // bS`.
- **Double Buffers:**
    - `k_scratch`: `(2, bS, D)`
    - `mask_scratch`: `(2, bT, bS)`
    - `score_scratch`: `(2, bT, bS)`
- **Pipeline Logic:**
    - Load `K[0]`, `Mask[0]`.
    - Loop `i`:
        - `curr = i % 2`, `next = (i+1) % 2`
        - Wait for `Score[curr]` write to finish (from previous usage).
        - Wait for `K[curr]`, `Mask[curr]` load.
        - Start load `K[next]`, `Mask[next]` (if not last).
        - Compute `Score[curr]`.
        - Start write `Score[curr]`.

### Backward QW Kernel (`backward_qw_kernel`)
- **Grid:** `(B, T // bT)`
- **Loop:** Iterate over `S // bS`.
- **Double Buffers:** `k_scratch` `(2, bS, D)`, `d_score_scratch` `(2, bT, bS)`.
- **Accumulators:** `d_q`, `d_w` in VMEM (no double buffering needed, they are accumulators).

### Backward K Kernel (`backward_k_kernel`)
- **Grid:** `(B, S // bS)`
- **Loop:** Iterate over `T // bT`.
- **Double Buffers:** `q_scratch` `(2, bT, H, D)`, `w_scratch` `(2, bT, H)`, `d_score_scratch` `(2, bT, bS)`.
- **Accumulator:** `d_k` in VMEM.

## 6. Expected Performance Impact
- **Throughput:** Expect 1.5x - 2x speedup due to hiding memory latency and better utilization of compute units with larger blocks.
- **Latency:** Reduced latency for large sequences.
- **Risks:**
    - Increased register pressure with larger blocks (unlikely to be an issue with these sizes).
    - Complexity of pipeline management (synchronization bugs).

## 7. Documentation Requirements
- **Shape Comments:** Explicitly state shapes of all tensors in the kernels.
- **Memory Space:** Annotate `VMEM` vs `HBM`.
- **Pipeline Comments:** Explain the prefetch and compute overlap logic.
- **Padding:** Document why padding is applied (TPU alignment).
