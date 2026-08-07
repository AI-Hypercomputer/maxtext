# Qwen3-80B Activation Offloading Worklog

## Activity Summary: Host Activation Offloading for Qwen3-80B
- **Date:** 2026-08-09
- **Model:** Qwen3-Next-80B (`qwen3-next-80b-a3b`)
- **Topology:** v6e-256 (256 TPU chips)
- **Target Mode:** `scan_layers=True` (`block_unroll=1`), `remat_policy=custom`
- **Key Results:**
  - Peak TPU HBM Memory: **30.04 GiB** (vs **397.90 GiB** baseline, **86% memory reduction**).
  - Offloaded Host Memory: **1.41 GiB** assigned to Pinned Host Memory (`pinned_host` `S(5)`).
  - AOT Compilation: **100% Successful** (`Finished train_compile.py successfully!`).
- **Configuration Specs:**
  - `decoder_layer_input=offload`, `context=device`
  - `block_unroll=1` and `skip_block_remat=False` in `src/maxtext/layers/nnx_decoders.py`
  - Identity offload isolation `inputs.astype(inputs.dtype)` in `src/maxtext/models/qwen3.py`
  - XLA Flags: `--xla_tpu_host_transfer_overlap_limit=1`, `--xla_tpu_aggressive_opt_barrier_removal=DISABLED`, `--xla_tpu_use_tc_device_shape_on_sc=false`
