---
name: performance_reviewer
description: >-
  Reviews and optimizes MaxText JAX/Flax code for TPU and GPU performance. Use when reviewing pull requests, implementing new model layers or attention mechanisms, or diagnosing throughput/memory bottlenecks in MaxText. Covers avoiding unintended host/CPU execution and synchronization, preventing full seq^2 attention matrix materialization in HBM in favor of fused kernels (Splash/Flash/Paged attention), using lax.scan over Python loops, preserving bfloat16/quantized compute dtypes, and maintaining proper GSPMD sharding and rematerialization annotations.
---

# MaxText Performance Review & Optimization Skill

When writing or reviewing MaxText code, evaluate changes against the following hardware-aware performance guidelines for TPUs and GPUs. Only flag an issue during code review when the diff introduces a verifiable performance bottleneck or anti-pattern, and provide a concrete, actionable recommendation.

---

## 1. Watch Out for Operations Running on CPU or Triggering Host-Device Synchronization

MaxText training and inference steps are compiled with `jax.jit` to execute asynchronously on TPU/GPU accelerators. Operations that execute on the host CPU or pull device tensors back to the host inside a JIT-compiled function, training step, or decode loop stall accelerator execution and break XLA fusion.

### What to Watch For
- **NumPy (`np.*`) on JAX Device Arrays**: Using `numpy` (`np.array`, `np.concatenate`, `np.where`, etc.) instead of `jax.numpy` (`jnp.*`) on traced or dynamic device tensors. Static shape/metadata calculations at trace time may use `np`, but tensor operations in the forward/backward/decode path must use `jnp` or `jax.lax`.
- **Implicit Host Transfers & Syncs**: Calling `jax.device_get()`, `.item()`, `np.asarray()`, `float(x)`, `int(x)`, `bool(x)`, or `print(x)` on device arrays inside the step loop or model forward pass.
- **Python Control Flow on Dynamic Tensors**: Using Python `if`, `while`, or `for` conditions that depend on the runtime value of a `jax.Array` (which forces host evaluation or fails under `jax.jit`) instead of `jnp.where`, `jax.lax.cond`, or `jax.lax.select`.
- **Host-Heavy Per-Step Preprocessing or Routing**: Performing token routing, sorting, mask construction, or metric aggregation on CPU every step when it can be expressed in JIT-compiled JAX (`jnp` / `jax.lax`) on device.
  - *Exception*: Offline checkpoint conversion scripts (e.g., `src/maxtext/checkpoint_conversion/`) intentionally load safetensors on CPU (`jax.default_device(jax.devices("cpu")[0])`) to avoid TPU HBM OOMs. Do not flag CPU usage in offline conversion utilities or unit test assertions.

### What to Recommend
- Replace `np.*` operations on dynamic tensors with `jnp.*` or `jax.lax.*` equivalents so XLA fuses them on the accelerator.
- Defer host transfers (`jax.device_get`, scalar extraction for logging) until outside the critical step loop or wrap asynchronous logging via `jax.debug.callback` / host callbacks only when needed.

---

## 2. Avoid Materializing Full seq^2 Matrices in HBM for Attention Mechanisms

Attention computations scale quadratically with sequence length (`L`). Materializing the full `[batch, heads, seq_len, seq_len]` (or `[seq_len, seq_len]`) attention score, mask, or softmax matrix in High-Bandwidth Memory (HBM) causes severe memory-bandwidth bottlenecks and Out-Of-Memory (OOM) errors at long context lengths.

### What to Watch For
- **Naive Dot-Product Attention in Hot Paths**: Computing full `QK^T` logits via `jnp.einsum` / `jnp.matmul` followed by `jax.nn.softmax` in HBM for standard global, causal, sliding-window, or chunked attention when a fused kernel is supported.
- **Materializing Dense `[L, L]` Attention Masks or Biases in HBM**: Constructing full 2D/4D boolean or float mask tensors (`jnp.tril(jnp.ones((seq_len, seq_len)))`, explicit `[B, 1, S, S]` causal/sliding masks, or dense relative position bias matrices) in HBM during training or prefill.
- **Custom Attention Variants Bypassing `AttentionOp`**: Implementing bespoke attention logic in decoder layers that materializes `[B, H, S, S]` intermediates instead of routing through MaxText's `Attention` / `AttentionOp` (`src/maxtext/layers/attentions.py` and `src/maxtext/layers/attention_op.py`).
- **Additive Masking Anti-Pattern**: When dot-product attention is legitimately used (e.g., small shape unit tests or unsupported fallback paths), adding a large negative mask tensor (`logits + mask`) instead of `jnp.where(mask >= DEFAULT_MASK_VALUE * 0.5, logits, DEFAULT_MASK_VALUE)` (see `apply_mask_to_logits` in `src/maxtext/layers/attention_op.py`), which can cause XLA to materialize the full masked tensor rather than fusing the predicate.

### What to Recommend
- **Use Fused Tiled Kernels**: Use MaxText's fused attention kernels (`attention="flash"` / Pallas & Tokamax Splash Attention on TPU, `cudnn_flash_te` on GPU, `paged_attention` / `ragged_attention` for inference, and Ring/Ulysses/USP kernels for Context Parallelism). These block Q/K/V into on-chip SRAM/VMEM tiles and perform online softmax accumulation without writing `[L, L]` intermediates to HBM.
- **Use Computable/Procedural Masks**: For Splash Attention on TPU, define masks using `splash_attention_mask._ComputableMask` subclasses (such as `ChunkedCausalMask`) or segment IDs so mask predicates are evaluated on-the-fly in VMEM/SMEM per tile rather than allocated in HBM.

---

## 3. Avoid Python Loops Over Layers or Sequence Positions

### What to Watch For
- **Unrolled Python Loops Over Decoder Layers or Sequence Steps**: Using `for layer in layers:` or `for t in range(seq_len):` inside JIT-compiled model execution when layers or steps share structure. Unrolling large loops in Python inflates the HLO graph, increases XLA compilation time, and prevents weight/instruction reuse.

### What to Recommend
- Use `jax.lax.scan` (e.g., MaxText's `scan_layers=True` block scanning pattern) for homogeneous decoder layer stacks or sequential recurrences, and use vectorized `jnp` / `jax.vmap` operations across batch/sequence dimensions.

---

## 4. Mixture-of-Experts (MoE) & Ragged Compute Efficiency

### What to Watch For
- **Per-Expert Python Loops or Heavily Padded Dense Matmuls**: Looping over experts in Python (`for expert_idx in range(num_experts):`) or zero-padding all experts to the maximum possible token capacity when grouped matrix multiplication is applicable.

### What to Recommend
- Route MoE computation through MaxText's grouped matrix multiplication kernels (Megablox GMM / `jax.lax.ragged_dot` / Tokamax GMM in `src/maxtext/layers/moe.py`) so only active token-expert assignments are computed on the MXU.

---

## 5. Precision, Dtypes, Sharding & Rematerialization

### What to Watch For
- **Unintended `float32` Upcasting on MXU/TensorCore Paths**: Mixing `float32` constants or auxiliary tensors with `bfloat16` activations before `einsum` / `dot_general` / `matmul` operations. JAX promotes `bfloat16` + `float32` to `float32`, doubling HBM bandwidth and halving MXU throughput.
  - *Note*: Explicit `float32` casting for numerically sensitive reductions (such as RMSNorm variance, RoPE frequencies, router softmax, or `float32_qk_product` / `float32_logits`) is intentional—ensure the result is cast back to `self.dtype` (`bfloat16`) before subsequent projections or GEMMs.
- **Missing Sharding Constraints or Remat Checkpoint Names**:
  - Omitting logical sharding annotations (`maybe_shard_with_logical`, `with_logical_constraint`) on new projections, KV caches, or large intermediate activations, which can lead GSPMD to pick suboptimal shardings or insert extra `all-gather` / `reduce-scatter` collectives.
  - Omitting `checkpoint_name(x, "...")` on major layer activations (e.g., `q_proj`, `k_proj`, `v_proj`, `out_proj`, `mlpwi_0`, `mlpwi_1`, `mlpwo`, `context`), which prevents MaxText's `remat_policy` and host-offloading configurations from managing activation memory.
