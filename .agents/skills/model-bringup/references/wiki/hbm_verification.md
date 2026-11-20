# Saving TPU HBM in Verification Runs

## Symptom / When to Use
Verification or training/SFT smoke checks exceed per-chip TPU HBM on either a single TPU VM or a multi-host TPU cluster (e.g., fitting a model onto a single TPU VM, or fitting a 512-chip model onto 64–128 chips). Verification prioritizes **fitting in HBM** over step throughput ("unoptimizing" MaxText).

## Action Ladder (Apply Stage-by-Stage)

### Stage 1: Zero-Code Config, Sharding & Offload Flags (Try First)
1. **Contiguous Layer-Scan Axis (`param_scan_axis=0`)**:
   - Default `param_scan_axis=1` places scanned blocks inside expert tensors (`[experts, blocks, ...]`), forcing XLA to allocate large transpose scratch buffers (~`+26.4 GiB/chip` on Qwen 3.8). Setting `param_scan_axis=0` (`[blocks, experts, ...]`) gives contiguous per-layer slices with zero copy overhead.
2. **Eliminate FSDP Expert All-Gather Buffers (`ici_fsdp_parallelism=1` with `EP × TP`)**:
   - When `EP` is supported by the model, set `ici_fsdp_parallelism=1` and shard experts across `ici_expert_parallelism × ici_tensor_parallelism` so each chip owns local experts without temporary FSDP all-gather buffers (~`-15.5 GiB/chip`).
3. **Offload or Drop Optimizer Moments**:
   - *Lossless AdamW*: Set `optimizer_memory_host_offload=true` and `mu_dtype=bfloat16` to park `mu`/`nu` in host CPU RAM until `apply_gradients` (~`-53–71 GiB/chip`).
   - *Stateless smoke runs*: Use `opt_type=sgd` when AdamW moment parity is not under test.
4. **Tile Vocab, Shrink Micro-Batch & Cap Ragged Buffers**:
   - Set `num_vocab_tiling=16` (tiles cross-entropy logits), fractional `per_device_batch_size=0.25` (decimates per-chip sequences), `ragged_buffer_factor=1.5` (bounds MoE dispatch buffers), `remat_policy=full`, `parameter_memory_host_offload=true`, and `LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536"`.

### Stage 2: Sub-Layer Remat & Temp-Buffer Reductions (If Activations / Clipping Spike)
- **Per-Sub-Layer Checkpointing**: Wrap attention and routed MoE sub-layers in individual `jax.checkpoint` boundaries inside scanned blocks (reduces live activation temp buffers to ~`1 GiB`) and stage weights per sub-layer.
- **Decoupled Vocab-Tiling Closure & 1-Copy Grad Clipping**: Keep full-layer intermediates out of vocab-tiling closures and scale gradients in-place during global-norm clipping to avoid duplicate gradient copies.

### Stage 3: Host-RAM `routed_experts` Streaming & In-Place Update Loop (Extreme Chip Squeeze)
- **Selective Host Streaming of `routed_experts`**: Stream routed expert weights and gradients between host DRAM and TPU HBM per block, with tunable partial projection offloading (e.g., offloading `8/12` vs `12/12` projections to trade host DRAM usage against TPU HBM).
- **Block-Streamed Grad Norms & In-Place `fori_loop` Updates**: Accumulate global grad norms across streamed blocks, apply optimizer updates via in-place `jax.lax.fori_loop`, and zero-initialize MoE ragged dispatch buffers.

### Reference Impact (Qwen 3.8 on TPU v5p, `per_device_batch_size=0.25`, `seq_len=4096`)

| Stage | Min v5p Chips | Configuration | Peak AOT HBM / Chip |
| :--- | :--- | :--- | :--- |
| Baseline | 512 | `adamw`, `param_scan_axis=1`, no offload | 58.52 GiB (OOM at $\le 256$ chips: 157.22 GiB) |
| Stage 1 (AdamW) | 256 | Lossless `adamw`, `param_scan_axis=0`, `EP×TP`, `optimizer_memory_host_offload=true` | 52.99 GiB |
| Stage 1 (SGD) | 128 | `sgd`, `param_scan_axis=0`, `EP×TP` | 90.65 GiB |
| Stage 2 | 128 | `sgd`, sub-layer remat + 1-copy grad clip | 90.12 GiB (~1.0 GiB act temp) |
| Stage 3A / 3B | 128 | Host-streamed projs + in-place `fori_loop` (`sgd` 12/12 / `adamw` 8/12) | 15.27 GiB (`sgd`) / 70.00 GiB (`adamw`) |
| Stage 3C | 64 | `sgd`, 8/12 projs host-streamed + zeroed ragged buffers | 68.57 GiB |

## Guardrails & Cleanup
- **Monitor Host DRAM**: Heavy host offloading (`optimizer_memory_host_offload`, `parameter_memory_host_offload`, Stage 3 streaming) shifts pressure to CPU RAM (e.g., ~`69–92 GiB/chip` host DRAM); tune partial offload ratios so hosts do not OOM.
- **Protect Default MFU & HLO CI**: Stage 1 CLI flags require no code changes. Any Stage 2/3 code modifications used solely to fit verification into HBM must either be gated behind an opt-in config flag (default `False` so `tests/integration/hlo_diff_test.py` and benchmark MFU are unaffected) or reverted during [pr_cleanup.md](../pr_cleanup.md) after archiving verification logs.
