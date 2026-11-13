# Pure-JAX Qwen3.5-397B-A17B Dual-SparseCore Batch-Split Trainer (64-Chip TPU v7x)

Standalone pure-JAX + Pallas Dual-SparseCore 2-microbatch batch-split trainer for
`Qwen3.5-397B-A17B` on a 64-chip TPU v7x (`tpu7x:4x4x4`, 128 TensorCores) slice.

## 64-Chip TPU v7x (`tpu7x:4x4x4`) Performance

Measured across `64` TPU v7x chips (`128` TensorCores, `2,097,152` tokens/step);
`eMFU` is reported against the TPU v7x BF16 peak (`2,307 TFLOPs/chip` = `1,153.5 TFLOPs/TC`)
using exact unpadded causal-attention, Gated DeltaNet recurrent/conv1d, and active top-`k`
MoE model FLOPs:

| Context Length | Recipe | Global Batch (`PDB`) | Mesh (`FSDP`, `EP=CP`) | Step Time | Throughput (`tok/s/chip`) | eMFU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64K** (`S=65,536`) | `recipes/qwen35_64c_base.sh` | `32` (`0.25`) | `FSDP=16, EP=8` | **`5.78 s`** (`5,775.7 ms`) | **`5,673.4`** | **`36.25%`** |
| **128K** (`S=131,072`) | `recipes/qwen35_64c_128k.sh` | `16` (`0.125`) | `FSDP=8, EP=16` | **`8.45 s`** (`8,445.8 ms`) | **`3,879.8`** | **`32.92%`** |
| **256K** (`S=262,144`) | `recipes/qwen35_64c_256k.sh` | `8` (`0.0625`) | `FSDP=4, EP=32` | **`9.53 s`** (`9,527.9 ms`) | **`3,439.2`** | **`43.59%`** |

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
*   **Dual-chunk load-balanced FP8 Q/K Splash Attention**: Full-attention layers
    permute Q-side activations into load-balanced dual-chunk order (rank `d`
    holds chunks `[d, 15-d]`) via two `ppermute`s, all-gather the 2 KV heads over
    CP, and execute a q-sharded Splash kernel with KV-grouped GQA sequence folding,
    FP8 (`e4m3fn`) Q/K quantization with prescaled `LOG2E`, and uniform-segment
    mask skipping (`ext_kernels/splash_attention/`).
*   **No-drop capacity-factor 3D FP8 MoE + Switch auxiliary loss**: Chunk 0 runs
    over the first aligned `capacity` sorted local-expert slots and checkpoints
    its activation bank, while any overflow slots execute in a `lax.while_loop`
    and rematerialize in the backward pass without dropping tokens (`gmmv2/`,
    `ext_kernels/ragged/`). Full-softmax probabilities and top-k counts accumulate
    across all 60 layers for the Qwen3.5 Switch load-balancing loss.
*   **Dual-SparseCore scheduling, Pallas VMEM RMSNorm & peeled-tail superblock**:
    EP dispatch/combine collectives are pinned to `SC0` while FSDP parameter
    all-gathers and gradient reduce-scatters run on `SC1` (`schedule.py`,
    `max_async_overlap_transform.py`). Dense weights automatically select
    `replicated`, `fsdp_only`, or `fsdp_expert` tiers based on
    `sc_min_collective_bytes` (`2 MB`), zero-centered RMSNorm runs in a custom
    Pallas VMEM kernel (`ext_kernels/rmsnorm_tc.py`), and the final superblock
    is peeled to keep all 4 MoE `gate_up` sublayers in HBM with zero PCIe
    host-offload stall at the forward-to-backward turn-around.

## Recipes (`recipes/`)

Ready-to-run shell recipes under `recipes/` inherit from `qwen35_64c_base.sh` and
override only the fields that differ:

*   `recipes/qwen35_64c_base.sh`: Canonical `64K` (`S=65,536`, `GBS=32`, `FSDP=16`, `EP=8`, `CF=1.25`).
*   `recipes/qwen35_64c_64k_pdb025_remat.sh`: `64K` aggressive HBM checkpointing (`save_moe_gate_up_sublayers=[0, 1, 3]`, halving PCIe offload).
*   `recipes/qwen35_64c_128k.sh`: `128K` context (`S=131,072`, `GBS=16`, `FSDP=8`, `EP=16`).
*   `recipes/qwen35_64c_256k.sh`: `256K` context (`S=262,144`, `GBS=8`, `FSDP=4`, `EP=32`).
*   `recipes/qwen35_64c_256k_pdb003125.sh`: `256K` zero-offload unsplit schedule (`S=262,144`, `GBS=4`, `PDB=1/32`, `FSDP=4`, `EP=32`, `schedule_mode=unsplit`).
*   `recipes/qwen35_64c_64k_pdb1.sh`: `64K` larger batch (`GBS=128`, `grad_accum_steps=2`, `sa_unroll_batch=true`).

Pass `--dry-run` to inspect resolved `CONFIG_OVERRIDES_JSON`, `--aot` to run
ahead-of-time compilation + HLO audit, or no flag on each TPU worker host to run
`benchmark_train`.

## Unit Tests (CPU)

Run the numerical parity, context-parallel GDN/Splash, no-drop capacity-factor
MoE, Switch load-balancing loss, parameter init scale, and batch-split gradient
equivalence suite on 8 simulated CPU devices:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=8" \
  pytest src/maxtext/experimental/qwen_35_64c/layers_test.py \
         src/maxtext/experimental/qwen_35_64c/splash_attention_test.py
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
