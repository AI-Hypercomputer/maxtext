# MaxText Review Focus Area: TPU and GPU Performance and Sharding

<!--
Shared focus area for MaxText code review. Used by the GitHub PR review
(.gemini/commands/maxtext-review.toml) and by an internal Code Review Agent
skill. Keep it platform-neutral.
-->

**Applies when the change touches:** `src/maxtext/layers/`,
`src/maxtext/models/`, `src/maxtext/m3/`, `src/maxtext/trainers/`,
`src/maxtext/training_engine/`, `src/maxtext/optimizers/`,
`src/maxtext/inference/`, `src/maxtext/multimodal/`,
`src/maxtext/utils/sharding.py`, `src/maxtext/utils/maxtext_utils.py`, or
`src/maxtext/utils/max_utils.py`.

MaxText training and inference steps compile with `jax.jit` to run
asynchronously across sharded TPU and GPU meshes. Keep a finding only if you can
name the function, tensor shape or dtype, and execution path (training step,
prefill, or decode) that triggers the bottleneck or failure, and provide a
concrete fix. Do not flag CPU usage or unrolled loops in offline scripts
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
    `jnp`/`jax.lax` on device. Also flag NumPy tensor ops (`np.array`,
    `np.concatenate`, `np.where`, etc.) on activations, logits, masks, or other
    per-step device arrays in the forward, backward, or decode path; recommend
    the `jnp`/`jax.lax` equivalent so XLA keeps them on device and fuses them.
    Flag Python `if`/`while`/`for` conditions or `range()` bounds that depend
    on the runtime value of a `jax.Array` (they fail under `jax.jit` or force a
    host sync outside it); recommend `jnp.where`, `jax.lax.select`,
    `jax.lax.cond`, `jax.lax.while_loop`, or `jax.lax.fori_loop`. Recommend
    keeping per-step metrics as on-device `jax.Array`s and buffering them across
    steps so step `t` is materialized on the host only after step `t + 1` is
    dispatched (e.g., `MetricLogger.buffer_and_write_metrics` in
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
    `ChunkedCausalMask`) or segment IDs; dense `[H, S, S]` or `[B, H, S, S]`
    relative-position or ALiBi-style bias tensors added to the logits; or
    bespoke decoder attention that bypasses `Attention`/`AttentionOp`
    (`src/maxtext/layers/attentions.py` and
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
7.  **Sharding changes that miss a path or a rule.** When a change replicates
    or re-shards a tensor's logical axis (e.g., keeping a single MQA K/V head
    replicated instead of splitting it over `tensor`), check every place that
    tensor and its cache are constrained, not only the training path: weight
    `kernel_axes`, activation constraints in training, prefill, and
    autoregressive modes, `shard_map` `in_specs` in attention kernels (flash
    attention builds its K/V specs from `flash_axis_names_kv`, which uses the
    query `HEAD` axis), dot-product attention's prefill/decode constraints, and
    the KV-cache logical axes (MLA keeps its single cache head unsharded with
    `CACHE_HEADS_NONE`). A dimension smaller than, or not divisible by, the
    product of its mesh axes fails at construction or trace time
    (`IndivisibleError`, or a `shard_map` divisibility `ValueError`). When a
    change adds a mesh axis or logical-axis rule, check every weight and
    activation rule that should include it, for every model family that uses
    those rules: a missing entry silently replicates the tensor over that axis
    instead of failing.
8.  **Unverified collective or memory claims.** When the description claims a
    change removes a collective (e.g., an all-gather of a TP-sharded
    activation) or saves memory, check what the claim depends on: the backward
    pass (a layout that avoids a forward all-gather can reintroduce it in the
    gradient), sibling modules that still flatten or reshape the sharded axis
    (the claim then holds only partially), and whether the default and
    model-specific `logical_axis_rules` shard that axis at all. Recommend a test
    that compiles a small sharded case (CPU CI provides 4 devices) and asserts
    on the collectives in the compiled HLO.
9.  **Rematerialization that does nothing.** `jax.checkpoint` (or `nnx.remat`)
    with `prevent_cse=False` is meant for use inside `jax.lax.scan`; elsewhere
    under `jit`, XLA can merge the recomputation with the forward pass and keep
    the activations, so the remat policy silently has no effect (e.g., on the
    `scan_layers=False` path). Derive the flag from
    `maxtext_utils.should_prevent_cse_in_remat(config)` instead of hardcoding
    it. Offloading policies cannot carry offloaded residuals across a
    trip-count-one loop; follow `_get_scan_safe_remat_policy` in
    `layers/nnx_decoders.py`, which keeps them on device there.
10. **Device-mesh construction.** Changes to `create_device_mesh`
    (`utils/maxtext_utils.py`) or the mesh helpers in `utils/max_utils.py`
    must handle chips that expose several devices (TPU7x and v2/v3 devices on
    one chip share `coords` and differ in `core_on_chip`; JAX's `mesh_utils`
    adds a core dimension for them), multi-slice and `use_lineage` meshes, and
    the other mesh options (`custom_mesh`, `allow_split_physical_axes`,
    `optimize_mesh_for_tpu_v6e`), which should be rejected or warned about
    rather than silently ignored. A layout that helps one mesh axis (e.g.,
    closing a ring on a slice dimension without wraparound links) changes the
    locality of the other axes, and wraparound differs by topology (see the
    wrap flags in `utils/accelerator_to_spec_map.py`): such an option can be
    strictly worse on a full torus, so its documentation must state its scope.
11. **Hand-built collectives under `shard_map`.** For
    `jax.lax.ragged_all_to_all`, `output_offsets` index into each *receiver's*
    output, and sender `i`'s `send_sizes[j]` must equal receiver `j`'s
    `recv_sizes[i]`; a custom VJP that returns gradients through the same
    collective must swap the send and receive sizes. Check that the offsets are
    built from the receiver's view and that both directions are derived from
    one consistent plan.
