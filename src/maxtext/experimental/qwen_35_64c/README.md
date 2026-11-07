# Pure-JAX Qwen3.5-397B-A17B Dual-SparseCore Batch-Split Trainer (64-Chip TPU v7x)

Standalone pure-JAX + Pallas Dual-SparseCore 2-microbatch batch-split trainer for
`Qwen3.5-397B-A17B` on a 64-chip TPU v7x (`tpu7x:4x4x4`, 128 TensorCores) slice.

## Architecture & Sharding Summary

| Setting | Default (`full_config()`) |
| :--- | :--- |
| Layers | `60` (`15` superblocks x `[GDN, GDN, GDN, FullAttn]`) |
| `emb_dim` / `vocab_size` | `4096` / `248,320` |
| Full Attention Q / KV heads (`head_dim`) | `32` / `2` (`256`, `partial_rotary_factor=0.25`) |
| Gated DeltaNet K / V heads (`head_dim`) | `16` / `64` (`128`, `conv_kernel=4`, `chunk_size=64`) |
| Routed / Shared MoE | `512` routed (`top_k=10`, `moe_mlp_dim=1024`) + `1` shared (`1024`) |
| Router load-balancing loss | Switch auxiliary loss (`router_aux_loss_coef=0.001`) |
| Global batch x Sequence length | `32 x 65,536` (`2,097,152` tokens/step, `grad_accum_steps=1`) |
| Physical torus mesh `('fsdp', 'ep_d2d', 'expert')` | `(16, 2, 4)` across `(y x x, core, z)` (`EP = CP = 8`, `64` local experts/EP rank) |
| Routed MoE precision & capacity | Lineage-style no-drop chunked 3D FP8 (`capacity_factor=1.25`) |

## Long Context (`64K` – `256K`) & Dual-SparseCore Overlap

The default configuration shards the global batch (`32`) across `fsdp=16` and the
`65,536`-token sequence across the EP group (`ep_d2d x expert` = `core x z`,
`CP = EP = 8`), yielding `2` sequences x `8,192` tokens per TensorCore so the two
batch-split microbatches correspond to the two local sequences:

*   **Gated DeltaNet CP scan**: Activations remain in natural contiguous sequence
    order (CP rank `r` holds chunks `[2r, 2r+1]` of 16) for cross-rank recurrent
    and conv1d state passing (`ext_kernels/gdn/`).
*   **Dual-chunk load-balanced Tokamax Splash Attention**: Full-attention layers
    permute Q-side activations into load-balanced dual-chunk order (rank `d`
    holds chunks `[d, 15-d]`) via two `ppermute`s, all-gather the 2 KV heads over
    CP, and execute a q-sharded Tokamax Splash kernel with GQA sequence folding.
*   **No-drop capacity-factor 3D FP8 MoE + Switch auxiliary loss**: Chunk 0 runs
    over the first aligned `capacity` sorted local-expert slots and checkpoints
    its activation bank, while any overflow slots execute in a `lax.while_loop`
    and rematerialize in the backward pass without dropping tokens (`gmmv2/`,
    `ext_kernels/ragged/`). Full-softmax probabilities and top-k counts accumulate
    across all 60 layers for the Qwen3.5 Switch load-balancing loss.
*   **Dual-SparseCore scheduling & `auto_sc_min` dense sharding**: EP
    dispatch/combine collectives are pinned to `SC0` while FSDP parameter
    all-gathers and gradient reduce-scatters run on `SC1` (`schedule.py`,
    `max_async_overlap_transform.py`). Dense weights automatically select
    `replicated`, `fsdp_only`, or `fsdp_expert` tiers based on
    `sc_min_collective_bytes` (`2 MB`) so sub-threshold collectives never fall
    back to synchronous TensorCore ICI, and chunked LM-head cross-entropy
    pipelines chunk `c+1`'s `SC1` weight gather behind chunk `c`'s MXU compute.

## Recipes (`recipes/`)

Ready-to-run shell recipes under `recipes/` inherit from `qwen35_64c_base.sh` and
override only the fields that differ:

*   `recipes/qwen35_64c_base.sh`: Canonical `64K` (`S=65,536`, `GBS=32`, `FSDP=16`, `EP=8`, `CF=1.25`).
*   `recipes/qwen35_64c_64k_pdb025_remat.sh`: `64K` aggressive HBM checkpointing (`save_moe_gate_up_sublayers=[0, 1, 3]`, halving PCIe offload).
*   `recipes/qwen35_64c_128k.sh`: `128K` context (`S=131,072`, `GBS=16`, `FSDP=8`, `EP=16`).
*   `recipes/qwen35_64c_256k.sh`: `256K` context (`S=262,144`, `GBS=8`, `FSDP=4`, `EP=32`).
*   `recipes/qwen35_64c_64k_pdb1.sh`: `64K` larger batch (`GBS=128`, `grad_accum_steps=2`, `sa_unroll_batch=true`).
*   `recipes/qwen35_64c_64k_pdb2.sh`: `64K` larger batch (`GBS=256`, `grad_accum_steps=4`, `sa_unroll_batch=true`).

Pass `--dry-run` to inspect resolved `CONFIG_OVERRIDES_JSON`, `--aot` to run
ahead-of-time compilation + HLO audit, or no flag on each TPU worker host to run
`benchmark_train`.

## Unit Tests (CPU)

Run the numerical parity, context-parallel GDN/Splash, no-drop capacity-factor
MoE, Switch load-balancing loss, parameter init scale, and batch-split gradient
equivalence suite on 8 simulated CPU devices:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=8" \
  pytest src/maxtext/experimental/qwen_35_64c/layers_test.py
```

## Ahead-of-Time (AOT) Compilation & HLO Audit

Compile the full 60-layer 64-chip `train_step` ahead of time for `tpu7x:4x4x4`
and emit memory + Dual-SparseCore collective/kernel audit statistics to
`aot_summary.json`:

```bash
python3 -m maxtext.experimental.qwen_35_64c.aot_compile \
  --exp_name=qwen35_397b_64c \
  --out_dir=/tmp/qwen35_aot \
  --topology=tpu7x:4x4x4
```

## Multi-Host Training on 64-Chip TPU v7x (`4x4x4`)

Run `benchmark_train` across all 16 hosts of a `tpu7x:4x4x4` slice (for example
via XPK, JobSet, or `gcloud compute tpus tpu-vm ssh --worker=all`):

```bash
# Synthetic data benchmark + JAX profiler capture on process 0:
python3 -m maxtext.experimental.qwen_35_64c.benchmark_train \
  --dataset_type=synthetic \
  --steps=18 \
  --warmup_steps=4 \
  --profile_steps=3 \
  --output_dir=gs://YOUR_BUCKET/qwen35_64c/synthetic

# Pre-tokenized C4 ArrayRecord training (65,536-token tf.train.Example rows):
python3 -m maxtext.experimental.qwen_35_64c.benchmark_train \
  --dataset_type=c4 \
  --c4_data_dir=gs://YOUR_BUCKET/c4_qwen35_s65536 \
  --steps=150 \
  --learning_rate=1e-4 \
  '--config_overrides_json={"lr_warmup_steps": 20, "grad_clip_norm": 1.0}' \
  --output_dir=gs://YOUR_BUCKET/qwen35_64c/c4_run
```
