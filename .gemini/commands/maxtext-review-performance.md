# MaxText Review Focus Area: TPU and GPU Performance

<!--
Shared focus area for MaxText code review. Used by the GitHub PR review
(.gemini/commands/maxtext-review.toml) and by an internal Code Review Agent
skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `src/maxtext/layers/`,
`src/maxtext/models/`, `src/maxtext/m3/`, `src/maxtext/trainers/`,
`src/maxtext/training_engine/`, `src/maxtext/optimizers/`,
`src/maxtext/inference/`, or `src/maxtext/multimodal/`.

MaxText training and inference steps compile with `jax.jit` to run
asynchronously across sharded TPU and GPU meshes. Keep a finding only if you can
name the function, tensor shape or dtype, and execution path (training step,
prefill, or decode) that triggers the bottleneck, and provide a concrete fix. Do
not flag CPU usage or unrolled loops in offline scripts
(`src/maxtext/checkpoint_conversion/`, `tools/`, `benchmarks/`) or unit tests.

## What to check

1.  **Host-device synchronization and CPU execution in hot paths.** Inside a
    JIT-compiled function, training step, or decode loop, flag operations that
    pull device arrays to the host or block async dispatch (`jax.device_get`,
    `jax.block_until_ready`, `.item()`, `np.asarray`, `float(x)`, `int(x)`,
    `bool(x)`, `print(x)`), host callbacks (`jax.pure_callback`,
    `jax.experimental.io_callback`, `jax.debug.callback`) in hot paths, or
    per-step token routing, sorting, mask construction, or metric aggregation
    performed on CPU every step when it can be expressed in JIT-compiled
    `jnp`/`jax.lax` on device. Recommend keeping per-step metrics as on-device
    `jax.Array`s and buffering them across steps so step `t` is materialized on
    the host only after step `t + 1` is dispatched (e.g.,
    `MetricLogger.buffer_and_write_metrics` in
    `src/maxtext/common/metric_logger.py`, `MetricsRecorder` in
    `src/maxtext/training_engine/metrics.py`, or N-step windowed
    `jax.device_get`). Do not flag `np.*` on static shapes or config constants
    evaluated once at trace time.
2.  **Materializing full `seq_len x seq_len` attention matrices in HBM.** Flag
    attention or mask implementations in training or prefill hot paths that
    materialize dense `[B, H, S, S]` or `[S, S]` score, softmax, or mask tensors
    in HBM: naive `QK^T` via `jnp.einsum`/`jnp.matmul` followed by
    `jax.nn.softmax` when a fused kernel applies (`attention="flash"` / Pallas
    and Tokamax Splash Attention on TPU, `cudnn_flash_te` on GPU,
    `paged_attention`/`ragged_attention` for inference, and Ring/Ulysses/USP for
    context parallelism); dense 2D/4D boolean or float causal/sliding masks
    (`jnp.tril`, `jnp.ones`) instead of procedural
    `splash_attention_mask._ComputableMask` subclasses (such as
    `ChunkedCausalMask`) or segment IDs; or bespoke decoder attention that
    bypasses `Attention`/`AttentionOp` (`src/maxtext/layers/attentions.py` and
    `src/maxtext/layers/attention_op.py`). When dot-product attention is
    legitimately used (e.g., fallback paths), flag additive masking
    (`logits + mask`) and recommend `apply_mask_to_logits` (`jnp.where`). Do not
    flag fixed `[..., chunk_size, chunk_size]` local blocks in chunked linear
    attention.
3.  **Unrolled Python loops over layers or sequence positions.** Flag decoder
    layer stacks that only loop in Python (`for layer in layers:`) without
    supporting `config.scan_layers` (`jax.lax.scan` / `nn.scan`) when layers are
    homogeneous or repeat in a regular cycle, or `for t in range(seq_len):`
    loops in training/prefill where `jax.lax.scan` or vectorized
    `jnp`/`jax.vmap` applies. Do not flag the `if not config.scan_layers:`
    branch when a scanned branch is also implemented.
4.  **MoE per-expert loops or dense padding.** Flag Python loops over experts
    (`for expert_idx in range(num_experts):`) or zero-padding all experts to the
    maximum token capacity when grouped matrix multiplication (Megablox GMM,
    `jax.lax.ragged_dot`, or Tokamax GMM in `src/maxtext/layers/moe.py`)
    applies.
5.  **Unintended `float32` promotion before GEMMs.** Numerically sensitive ops
    (RMSNorm variance, RoPE frequencies, router softmax, or explicit
    `float32_qk_product`/`float32_logits`) intentionally cast to `float32`; flag
    cases where the result is not cast back to `self.dtype` (`bfloat16`) before
    subsequent `DenseGeneral`, `einsum`, `dot_general`, or attention calls, or
    where a `jnp.float32` tensor promotes a `bfloat16` activation prior to a
    matmul. (Python `float` scalars are weakly typed in JAX and do not promote
    `bfloat16`.)
6.  **Missing or misapplied sharding and rematerialization annotations.** In new
    model layers or projections, flag missing logical sharding constraints
    (`maybe_shard_with_logical` from `maxtext.utils.sharding`, or
    `sharding.constrain` in `src/maxtext/m3/`) on projections, KV caches, or
    large intermediates. Prefer `maybe_shard_with_logical` whenever `mesh` and
    `shard_mode` are available, and flag `with_logical_constraint` on KV
    caches, inference-reachable layers (`MaxEngine` enters its mesh via
    `with mesh:`, where `with_logical_constraint` is a silent no-op), or
    `ShardMode.EXPLICIT` paths (where `with_logical_constraint` does not
    dispatch to `reshard`). Also flag missing `checkpoint_name(x, "...")` tags
    on major activations using the exact names recognized by `remat_policy`
    (`query_proj`, `key_proj`, `value_proj`, `qkv_proj`, `kv_proj`, `out_proj`,
    `context`, `decoder_layer_input`, `mlpwi_0`, `mlpwi_1`, `mlpwi`, `mlpwo`,
    `moe_mlpwi_0`, `moe_mlpwi_1`, `moe_mlpwo`).
