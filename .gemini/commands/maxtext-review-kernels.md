# MaxText Review Focus Area: TPU Kernels

<!--
Shared focus area for MaxText code review. Used by the GitHub PR review
(.gemini/commands/maxtext-review.toml) and by an internal Code Review Agent
skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `src/maxtext/kernels/` (megablox grouped
matmul, splash/ragged/ring attention, gather/reduce/sort, mhc).

MaxText kernels are Python Pallas kernels (`pl.pallas_call` with
`jax.experimental.pallas.tpu` as `pltpu`), SparseCore kernels, and wrappers
around tokamax kernels. Kernel bugs usually do not crash: they produce slightly
wrong numbers, wrong gradients, or out-of-bounds reads that only show up for
some shapes, group sizes, or TPU generations. Review the changed kernel code,
its callers that select or configure the kernel, and the kernel tests in
`tests/unit/`. Keep a finding only if you can name the shape, group-size
distribution, dtype, or TPU generation that triggers it and what goes wrong. Do
not speculate about performance without a concrete mechanism (e.g., a block
that exceeds VMEM, a serialized DMA, a dimension marked `ARBITRARY` that could
be `PARALLEL`).

## What to check

1.  **Grid, BlockSpecs, and index maps.** Index maps must return block indices
    (not element offsets) and match the grid's dimension order. Check that
    every input/output block is covered exactly once when the kernel writes
    outputs, that `PrefetchScalarGridSpec` scalar-prefetch arguments (e.g.,
    group offsets/sizes for ragged or grouped matmul) are indexed consistently
    in the index maps and the kernel body, and that revisited output blocks
    are initialized and accumulated correctly.
2.  **Ragged and non-divisible shapes.** Dimensions that are not multiples of
    the block size, empty groups (size 0), groups that straddle block
    boundaries, and the last partial block must be padded or masked. Masks
    must cover both the reads and the writes, and padding values must not leak
    into reductions (e.g., `-inf` vs `0` for max vs sum, softmax denominators).
3.  **Tiling and VMEM.** Block shapes must respect TPU layout constraints
    (last two dimensions aligned to the (sublane, lane) tiling for the dtype)
    and, together with double buffering and scratch buffers, fit in VMEM.
    Check `vmem_limit_bytes` in `pltpu.CompilerParams` against the real
    footprint, and prefer deriving limits and tile sizes from
    `pltpu.get_tpu_info()` over hardcoding values that only fit one generation.
4.  **DMA and synchronization.** Every `pltpu.make_async_copy` `start()` must
    be paired with `wait()` on the same semaphore before the destination is
    read or the source buffer is reused; check buffer slot indexing in manual
    double-buffering and `pltpu.emit_pipeline`, and semaphore counts across
    loop iterations. On SparseCore and TensorCore/SparseCore hand-offs, check
    that producers and consumers are synchronized.
5.  **Dimension semantics.** Grid dimensions marked `PARALLEL` must have no
    cross-iteration dependency (no accumulation into the same output block, no
    carried scratch state); dimensions that carry state must be `ARBITRARY`.
    A wrong `PARALLEL` marking is a silent race on multi-core chips.
6.  **Numerics.** Accumulate in float32 (`preferred_element_type`) when inputs
    are bf16/fp8/int8; check scaling and dequantization order for quantized
    paths, online-softmax rescaling in attention kernels, and causal or
    segment masks.
7.  **Custom VJP.** For `jax.custom_vjp` kernels, the backward pass must
    compute the gradient of the forward that is actually used (same masking,
    scaling, and group layout), save only the residuals it needs, and return
    gradients with the correct shapes and dtypes for every primal input
    (including `None`/zeros for non-differentiable ones).
8.  **Integration and fallback.** Callers that choose between kernel and
    reference paths (via config flags or hardware checks) must not silently
    fall back, change the default, or pass different shapes, shardings, or
    dtypes to the kernel than the reference path receives. Under `shard_map`,
    check that per-shard shapes match the kernel's assumptions.
9.  **Tests.** A new or changed kernel needs a test in `tests/unit/` that
    compares outputs (and gradients, for custom VJP) against a pure-JAX
    reference with dtype-appropriate tolerances, over shapes that exercise
    non-divisible dimensions, empty and uneven groups, and the masking path.
    Tests that run only in `interpret=True` mode are fine for correctness but
    do not catch VMEM or layout problems; flag tolerances loose enough to hide
    an indexing bug, and tests that compare the kernel against itself.
